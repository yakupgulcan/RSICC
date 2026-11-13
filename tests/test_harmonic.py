import unittest

try:
    import numpy as np
    import torch
except ImportError:
    raise unittest.SkipTest('Requires the training environment with NumPy and PyTorch.')

from checkpoints import build_models
from harmonic import harmonic_smooth, shared_laplacian


def reference_structural(feat1, feat2, alpha):
    """Direct port of the MATLAB demo for one pair of (c, h, w) arrays, with explicit edge lists."""
    c, h, w = feat1.shape
    n = h * w
    x1, x2 = feat1.reshape(c, n).T, feat2.reshape(c, n).T
    z = 0.5 * (x1 + x2)
    z = (z - z.mean(axis=0)) / (z.std(axis=0, ddof=1) + 1e-6)
    idx = np.arange(n).reshape(h, w)
    pairs = np.concatenate([
        np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1),
        np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1),
        np.stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel()], axis=1),
        np.stack([idx[1:, :-1].ravel(), idx[:-1, 1:].ravel()], axis=1),
    ])
    delta2 = ((z[pairs[:, 0]] - z[pairs[:, 1]]) ** 2).sum(axis=1)
    # An odd edge count would make NumPy's median one element; this grid has an even count,
    # where PyTorch takes the lower middle element.
    sigma2 = max(np.sort(delta2)[(len(delta2) - 1) // 2], 1e-6)
    weights = np.exp(-delta2 / (2 * sigma2))
    affinity = np.zeros((n, n))
    affinity[pairs[:, 0], pairs[:, 1]] = weights
    affinity[pairs[:, 1], pairs[:, 0]] = weights
    scale = 1 / np.sqrt(np.maximum(affinity.sum(axis=1), 1e-8))
    laplacian = np.eye(n) - scale[:, None] * affinity * scale[None, :]
    system = np.eye(n) + alpha * laplacian
    return [np.linalg.solve(system, x).T.reshape(c, h, w) for x in (x1, x2)]


class HarmonicSmoothTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.feat1 = torch.rand(2, 5, 6, 7, dtype=torch.float64)
        self.feat2 = torch.rand(2, 5, 6, 7, dtype=torch.float64)

    def test_zero_alpha_returns_the_inputs(self):
        out1, out2 = harmonic_smooth(self.feat1, self.feat2, 0.0)
        self.assertIs(out1, self.feat1)
        self.assertIs(out2, self.feat2)

    def test_matches_the_reference_implementation(self):
        for alpha in (0.5, 8.0):
            out1, out2 = harmonic_smooth(self.feat1, self.feat2, alpha)
            for item in range(2):
                ref1, ref2 = reference_structural(self.feat1[item].numpy(), self.feat2[item].numpy(), alpha)
                np.testing.assert_allclose(out1[item].numpy(), ref1, atol=1e-9)
                np.testing.assert_allclose(out2[item].numpy(), ref2, atol=1e-9)

    def test_laplacian_is_symmetric_with_eight_neighbours(self):
        laplacian = shared_laplacian(self.feat1)
        self.assertTrue(torch.allclose(laplacian, laplacian.transpose(1, 2)))
        # Off-diagonal entries: two per edge (horizontal, vertical and both diagonals of a 6x7 grid).
        self.assertEqual(int((laplacian[0] != 0).sum()) - 6 * 7, 2 * (5 * 7 + 6 * 6 + 2 * 5 * 6))

    def test_smoothing_reduces_variation_and_is_differentiable(self):
        feat1 = self.feat1.clone().requires_grad_()
        out1, out2 = harmonic_smooth(feat1, self.feat2, 2.0)
        self.assertLess(out1.var(dim=(2, 3)).sum().item(), self.feat1.var(dim=(2, 3)).sum().item())
        (out1.sum() + out2.sum()).backward()
        self.assertTrue(torch.isfinite(feat1.grad).all())


class ModelIntegrationTests(unittest.TestCase):
    settings = {'encoder_image': 'resnet101', 'encoder_feat': 'MCCFormers_diff_as_Q', 'decoder': 'trans',
                'n_heads': 8, 'n_layers': 1, 'decoder_n_layers': 1, 'dropout': 0.5,
                'feature_dim_de': 1024, 'vocab_size': 10}

    def encoder_feat(self, **extra):
        from unittest.mock import patch
        with patch('models.CNN_Encoder'):  # skip the pretrained-weights download
            return build_models({**self.settings, **extra})[1]

    def test_settings_without_alpha_build_the_baseline(self):
        self.assertEqual(self.encoder_feat().harmonic_alpha, 0.0)

    def test_alpha_changes_the_output_without_adding_weights(self):
        import models_RSICCformerDfusion
        models_RSICCformerDfusion.device = torch.device('cpu')
        baseline, smoothed = self.encoder_feat().eval(), self.encoder_feat(harmonic_alpha=2.0).eval()
        self.assertEqual(baseline.state_dict().keys(), smoothed.state_dict().keys())
        smoothed.load_state_dict(baseline.state_dict())
        feat1, feat2 = torch.rand(2, 1024, 14, 14), torch.rand(2, 1024, 14, 14)
        with torch.no_grad():
            out_baseline, out_smoothed = baseline(feat1, feat2), smoothed(feat1, feat2)
        self.assertEqual(out_baseline.shape, out_smoothed.shape)
        self.assertFalse(torch.allclose(out_baseline, out_smoothed))


if __name__ == '__main__':
    unittest.main()
