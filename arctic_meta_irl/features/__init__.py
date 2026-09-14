"""State-feature construction phi(s) for reward learning.

Public API lives in :mod:`arctic_meta_irl.features.builder`; both
``from arctic_meta_irl.features import FeatureBuilder`` and
``from arctic_meta_irl.features.builder import FeatureBuilder`` work.
"""
from .builder import FeatureBuilder, FeatureSpec

__all__ = ["FeatureBuilder", "FeatureSpec"]
