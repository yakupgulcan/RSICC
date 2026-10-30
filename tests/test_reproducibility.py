import os
import random
import unittest
from unittest.mock import patch

from reproducibility import configure_randomness, resolve_seed, seed_worker


class SeedSelectionTests(unittest.TestCase):
    def test_fixed_and_random_selection(self):
        self.assertEqual(resolve_seed(123), 123)
        with patch('reproducibility.secrets.randbits', return_value=456):
            self.assertEqual(resolve_seed(None), 456)

    def test_invalid_seeds_are_rejected(self):
        for value in (-1, 2**32, True, 1.5, '123'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_seed(value)


class RandomnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy
            import torch
        except ImportError:
            raise unittest.SkipTest('Requires the training environment with NumPy and PyTorch.')
        cls.np = numpy
        cls.torch = torch

    def setUp(self):
        self.python_state = random.getstate()
        self.numpy_state = self.np.random.get_state()
        self.torch_state = self.torch.get_rng_state()
        self.benchmark = self.torch.backends.cudnn.benchmark
        self.deterministic = self.torch.backends.cudnn.deterministic
        self.algorithms = self.torch.are_deterministic_algorithms_enabled()
        self.environment = patch.dict(os.environ)
        self.environment.start()

    def tearDown(self):
        random.setstate(self.python_state)
        self.np.random.set_state(self.numpy_state)
        self.torch.set_rng_state(self.torch_state)
        self.torch.backends.cudnn.benchmark = self.benchmark
        self.torch.backends.cudnn.deterministic = self.deterministic
        self.torch.use_deterministic_algorithms(self.algorithms)
        self.environment.stop()

    def draws(self):
        return (random.random(), self.np.random.random(), self.torch.rand(1).item(),
                self.torch.randperm(20).tolist())

    def test_same_seed_repeats_and_another_seed_changes_streams(self):
        configure_randomness(123)
        first = self.draws()
        configure_randomness(123)
        self.assertEqual(first, self.draws())
        configure_randomness(124)
        self.assertNotEqual(first, self.draws())

    def test_deterministic_mode_can_be_enabled_and_disabled(self):
        configure_randomness(123, deterministic=True)
        self.assertFalse(self.torch.backends.cudnn.benchmark)
        self.assertTrue(self.torch.are_deterministic_algorithms_enabled())
        self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        configure_randomness(124, deterministic=False)
        self.assertTrue(self.torch.backends.cudnn.benchmark)
        self.assertFalse(self.torch.are_deterministic_algorithms_enabled())

    def test_worker_python_and_numpy_seeds_follow_torch_worker_seed(self):
        with patch.object(self.torch, 'initial_seed', return_value=2**32 + 123):
            seed_worker(0)
        observed = random.random(), self.np.random.random()
        random.seed(123)
        self.np.random.seed(123)
        self.assertEqual(observed, (random.random(), self.np.random.random()))


if __name__ == '__main__':
    unittest.main()
