"""
EpistemX - Earth Observation Data Processing Package

A comprehensive package for processing and analyzing Earth observation data
using Google Earth Engine and other remote sensing tools.
"""

from .ee_config import (
    initialize_earth_engine, 
    ensure_ee_initialized, 
    is_ee_initialized,
    initialize_with_service_account,
    authenticate_manually,
    get_auth_status,
    print_auth_instructions,
    setup_earth_engine,
    reset_ee_initialization
)
import os
import warnings

# Do not automatically initialize Earth Engine at import time
# Users should call setup_earth_engine() or ensure_ee_initialized() when needed

# Check for service account file in environment or common locations
def _find_service_account_file():
    """Look for service account file in common locations."""
    # Check environment variable first
    env_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if env_file and os.path.exists(env_file):
        return env_file
    
    # Check auth/ directory first (project-specific location)
    # Use absolute path to ensure it works regardless of working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up from src/epistemx to project root
    auth_dir = os.path.join(project_root, 'auth')
    
    if os.path.exists(auth_dir):
        for file in os.listdir(auth_dir):
            if file.endswith('.json'):
                auth_file = os.path.join(auth_dir, file)
                if os.path.exists(auth_file):
                    return auth_file
    
    # Also check relative auth/ directory (fallback)
    auth_dir_relative = 'auth'
    if os.path.exists(auth_dir_relative):
        for file in os.listdir(auth_dir_relative):
            if file.endswith('.json'):
                auth_file = os.path.join(auth_dir_relative, file)
                if os.path.exists(auth_file):
                    return auth_file
    
    # Check common file names in current directory
    common_names = [
        'service-account.json',
        'gee-service-account.json',
        'earth-engine-service-account.json',
        'credentials.json'
    ]
    
    for name in common_names:
        if os.path.exists(name):
            return name
    
    return None

# Helper function for manual initialization (can be called by users)
def auto_initialize():
    """Attempt automatic initialization with fallback options."""
    try:
        # First, try to find and use service account
        service_account_file = _find_service_account_file()
        if service_account_file:
            if setup_earth_engine(service_account_file=service_account_file):
                print(f"Earth Engine initialized with service account: {service_account_file}")
                return True
        
        # If no service account, try regular initialization
        if setup_earth_engine():
            print("Earth Engine initialized successfully")
            return True
        
        return False
        
    except Exception as e:
        warnings.warn(
            f"Could not automatically initialize Earth Engine: {e}. "
            "You may need to authenticate manually. Use print_auth_instructions() for help.",
            UserWarning
        )
        return False

__version__ = "0.1.0"
__author__ = "EpistemX Team"

# Make key functions available at package level
__all__ = [
    'initialize_earth_engine',
    'ensure_ee_initialized', 
    'is_ee_initialized',
    'initialize_with_service_account',
    'authenticate_manually',
    'get_auth_status',
    'print_auth_instructions',
    'setup_earth_engine',
    'reset_ee_initialization',
    'auto_initialize'
]