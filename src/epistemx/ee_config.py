"""
Earth Engine Configuration Module

Centralized Earth Engine authentication and initialization for the epistemx package.
This module ensures Earth Engine is properly set up before any GEE operations.
"""

import ee
import logging
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

# Global flag to track initialization status
_ee_initialized = False

def initialize_earth_engine(project: Optional[str] = None, force_reinit: bool = False) -> bool:
    """
    Initialize Google Earth Engine with authentication.
    
    Parameters
    ----------
    project : str, optional
        GEE project ID. If None, uses default project.
    force_reinit : bool, default False
        Force re-initialization even if already initialized.
        
    Returns
    -------
    bool
        True if initialization successful, False otherwise.
        
    Example
    -------
    >>> from epistemx.ee_config import initialize_earth_engine
    >>> initialize_earth_engine()
    """
    global _ee_initialized
    
    if _ee_initialized and not force_reinit:
        logger.debug("Earth Engine already initialized")
        return True
    
    try:
        # Try to initialize without authentication first (for already authenticated users)
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        
        _ee_initialized = True
        logger.info("Earth Engine initialized successfully")
        return True
        
    except ee.EEException as e:
        if "not authenticated" in str(e).lower():
            logger.info("Earth Engine authentication required")
            try:
                # Authenticate and then initialize
                ee.Authenticate()
                if project:
                    ee.Initialize(project=project)
                else:
                    ee.Initialize()
                
                _ee_initialized = True
                logger.info("Earth Engine authenticated and initialized successfully")
                return True
                
            except Exception as auth_error:
                logger.error(f"Earth Engine authentication failed: {auth_error}")
                return False
        else:
            logger.error(f"Earth Engine initialization failed: {e}")
            return False
    
    except Exception as e:
        logger.error(f"Unexpected error during Earth Engine initialization: {e}")
        return False

def ensure_ee_initialized(project: Optional[str] = None) -> None:
    """
    Ensure Earth Engine is initialized, raising an exception if it fails.
    
    Parameters
    ----------
    project : str, optional
        GEE project ID. If None, uses default project.
        
    Raises
    ------
    RuntimeError
        If Earth Engine initialization fails.
    """
    if not initialize_earth_engine(project=project):
        raise RuntimeError(
            "Failed to initialize Google Earth Engine. "
            "Please check your authentication and internet connection."
        )

def is_ee_initialized() -> bool:
    """
    Check if Earth Engine is initialized.
    
    Returns
    -------
    bool
        True if Earth Engine is initialized, False otherwise.
    """
    return _ee_initialized

def reset_ee_initialization() -> None:
    """
    Reset the initialization flag. Useful for testing or troubleshooting.
    """
    global _ee_initialized
    _ee_initialized = False
    logger.debug("Earth Engine initialization flag reset")