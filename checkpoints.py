"""Model construction and checkpoints stored as weights plus the settings that built them.

Weights-only checkpoints stay loadable when model code is edited, and loading
fails loudly when the architecture no longer matches. Checkpoints written before
this format hold pickled model objects; they can still be evaluated.
"""
import torch

MODEL_SETTINGS = ('encoder_image', 'encoder_feat', 'decoder', 'n_heads', 'n_layers',
                  'decoder_n_layers', 'dropout', 'feature_dim_de', 'harmonic_alpha')
MODELS = ('encoder_image', 'encoder_feat', 'decoder')


# Settings added after checkpoints already existed; older checkpoints mean these values.
LATER_SETTINGS = {'harmonic_alpha': 0.0}


def model_settings(args, vocab_size):
    settings = {name: getattr(args, name) for name in MODEL_SETTINGS}
    settings['vocab_size'] = vocab_size
    return settings


def build_models(settings):
    settings = {**LATER_SETTINGS, **settings}
    from models import CNN_Encoder
    from models_RSICCformerDfusion import DecoderTransformer, MCCFormers_diff_as_Q

    if settings['encoder_feat'] != 'MCCFormers_diff_as_Q':
        raise ValueError(f"Unknown encoder_feat: {settings['encoder_feat']}")
    if settings['decoder'] != 'trans':
        raise ValueError(f"Unknown decoder: {settings['decoder']}")
    encoder_image = CNN_Encoder(NetType=settings['encoder_image'], method=settings['decoder'])
    # The feature encoder's dropout of 0.5 is the original training setting.
    encoder_feat = MCCFormers_diff_as_Q(feature_dim=1024, dropout=0.5, h=14, w=14, d_model=512,
                                        n_head=settings['n_heads'], n_layers=settings['n_layers'],
                                        harmonic_alpha=settings['harmonic_alpha'])
    decoder = DecoderTransformer(feature_dim=settings['feature_dim_de'],
                                 vocab_size=settings['vocab_size'],
                                 n_head=settings['n_heads'],
                                 n_layers=settings['decoder_n_layers'],
                                 dropout=settings['dropout'])
    return encoder_image, encoder_feat, decoder


def state_dicts(objects):
    return {name: obj.state_dict() if obj is not None else None for name, obj in objects.items()}


def read_checkpoint(path, device='cpu'):
    """Return (checkpoint, is_legacy). Legacy files need unpickling, so load only trusted ones."""
    try:
        return torch.load(path, map_location=device, weights_only=True), False
    except Exception:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if 'model_settings' in checkpoint:
            raise
        return checkpoint, True


def load_models(path, device):
    """Models for inference from a current or legacy checkpoint."""
    checkpoint, legacy = read_checkpoint(path, device)
    if legacy:
        models = [checkpoint[name] for name in MODELS]
    else:
        models = build_models(checkpoint['model_settings'])
        for name, model in zip(MODELS, models):
            model.load_state_dict(checkpoint['models'][name])
    return [model.to(device) for model in models]
