"""Shared seed controls for preprocessing, training, and inference."""
import os
import random
import secrets


def resolve_seed(seed):
    if seed is None:
        return secrets.randbits(32)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('Seed must be None or an integer in [0, 2**32).')
    return seed


def add_seed_arguments(parser, default=None):
    parser.add_argument('--seed', type=int, default=default,
                        help='random seed; omitted means unseeded (preprocessing defaults to 123).')
    parser.add_argument('--deterministic', action='store_true',
                        help='require deterministic PyTorch algorithms; unsupported operations raise an error.')


def configure_randomness(seed, deterministic=False):
    import numpy as np
    import torch

    if seed is not None:
        resolve_seed(seed)  # Validate without selecting a different seed.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id):
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
