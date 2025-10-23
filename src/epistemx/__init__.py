"""
EpistemX - Earth Observation Data Processing Package

A comprehensive package for processing and analyzing Earth observation data
using Google Earth Engine and other remote sensing tools.
"""

from .ee_config import initialize_earth_engine, ensure_ee_initialized, is_ee_initialized

# Automatically initialize Earth Engine when package is imported
try:
    initialize_earth_engine()
except Exception as e:
    import warnings
    warnings.warn(
        f"Could not automatically initialize Earth Engine: {e}. "
        "You may need to call initialize_earth_engine() manually.",
        UserWarning
    )

__version__ = "0.1.0"
__author__ = "EpistemX Team"

# Make key functions available at package level
__all__ = [
    'initialize_earth_engine',
    'ensure_ee_initialized', 
    'is_ee_initialized'
]