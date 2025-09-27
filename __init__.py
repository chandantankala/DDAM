"""Dense Associative Memory on the Bures-Wasserstein Space"""

from .core import DDAM
from .algorithms import phi_operator, retrieve
from .metrics import wasserstein_distance_gaussian, bures_wasserstein_distance
from .utils import (
    sample_gaussian_sphere,
    sample_commuting_gaussians,
    perturb_gaussian
)

__version__ = '0.1.0'
__all__ = [
    'DDAM',
    'phi_operator',
    'retrieve',
    'wasserstein_distance_gaussian',
    'bures_wasserstein_distance',
    'sample_gaussian_sphere',
    'sample_commuting_gaussians',
    'perturb_gaussian'
]
