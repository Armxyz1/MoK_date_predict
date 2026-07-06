"""
Normalization statistics computation and management.

This module provides functionality to compute, save, and load
normalization statistics (mean and std) from training data.
"""

from .compute_stats import compute_normalization_stats, NormalizationStats
from .stats_manager import save_normalization_stats, load_normalization_stats, stats_exist

__all__ = [
    'compute_normalization_stats',
    'NormalizationStats',
    'save_normalization_stats',
    'load_normalization_stats',
    'stats_exist',
]
