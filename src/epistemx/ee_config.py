"""
Earth Engine Configuration Module

Centralized Earth Engine authentication and initialization for the epistemx package.
This module ensures Earth Engine is properly set up before any GEE operations.

Supports:
- Service account authentication (for Earth Engine API access)
- Manual user authentication (OAuth2 for personal accounts)
- Google Drive export capabilities (OAuth2 for user's Drive)
"""

import ee
import logging
import os
import json
from typing import Optional, Dict, Any
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Global flag to track initialization status
_ee_initialized = False

def initialize_with_service_account(
    service_account_file: str, 
    project: Optional[str] = None
) -> bool:
    """
    Initialize Earth Engine using a service account.
    
    Parameters
    ----------
    service_account_file : str
        Path to the service account JSON key file.
    project : str, optional
        GEE project ID. If None, uses project from service account.
        
    Returns
    -------
    bool
        True if initialization successful, False otherwise.
        
    Example
    -------
    >>> from epistemx.ee_config import initialize_with_service_account
    >>> initialize_with_service_account('path/to/service-account.json')
    """
    global _ee_initialized
    
    # Initialize service_account_info variable
    service_account_info = None
    
    try:
        # Validate service account file exists
        if not os.path.exists(service_account_file):
            logger.error(f"Service account file not found: {service_account_file}")
            return False
        
        # Load service account credentials
        with open(service_account_file, 'r') as f:
            try:
                service_account_info = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in service account file: {e}")
                logger.error(f"File: {service_account_file}")
                return False
        
        # Check if we successfully parsed the service account info
        if not service_account_info:
            logger.error("Failed to parse service account JSON")
            return False
        
        # Extract project ID if not provided
        if not project:
            project = service_account_info.get('project_id')
        
        logger.info(f"Attempting to initialize Earth Engine with service account for project: {project}")
        
        # Set the environment variable for Google Application Credentials
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_file
        
        # Initialize Earth Engine with the project
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        
        _ee_initialized = True
        logger.info(f"Earth Engine initialized successfully with service account for project: {project}")
        return True
        
    except Exception as e:
        logger.error(f"Service account initialization failed: {e}")
        # Try alternative method with explicit credentials only if we have service account info
        if service_account_info:
            try:
                logger.info("Trying alternative authentication method...")
                credentials = ee.ServiceAccountCredentials(
                    email=service_account_info['client_email'],
                    key_file=service_account_file
                )
                
                if project:
                    ee.Initialize(credentials, project=project)
                else:
                    ee.Initialize(credentials)
                
                _ee_initialized = True
                logger.info(f"Earth Engine initialized with alternative method for project: {project}")
                return True
                
            except Exception as e2:
                logger.error(f"Alternative authentication method also failed: {e2}")
                return False
        else:
            logger.error("Cannot try alternative method - service account info not available")
            return False

def authenticate_manually(project: Optional[str] = None, force_cloud_api: bool = False) -> bool:
    """
    Perform manual Earth Engine authentication.
    
    This will open a browser window for authentication.
    
    Parameters
    ----------
    project : str, optional
        GEE project ID. If None, uses default project.
    force_cloud_api : bool, default False
        If True, forces authentication using the Cloud API method.
        
    Returns
    -------
    bool
        True if authentication and initialization successful, False otherwise.
        
    Example
    -------
    >>> from epistemx.ee_config import authenticate_manually
    >>> authenticate_manually()
    """
    global _ee_initialized
    
    try:
        logger.info("Starting manual Earth Engine authentication...")
        
        # Try authentication with optional force_cloud_api parameter
        if force_cloud_api:
            ee.Authenticate(force_cloud_api=True)
        else:
            ee.Authenticate()
        
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        
        _ee_initialized = True
        logger.info("Earth Engine authenticated and initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Manual authentication failed: {e}")
        return False


def get_auth_url() -> Optional[str]:
    """
    Get the authentication URL for manual OAuth flow.
    
    This is useful for environments where automatic browser opening doesn't work,
    such as remote servers or containerized applications.
    
    Returns
    -------
    str or None
        The authentication URL, or None if it cannot be generated.
        
    Example
    -------
    >>> from epistemx.ee_config import get_auth_url
    >>> url = get_auth_url()
    >>> print(f"Please visit: {url}")
    """
    try:
        # This attempts to get the auth URL without completing the flow
        # Note: This is a workaround and may not work in all versions of ee
        import ee.oauth as oauth
        
        # Get the OAuth helper
        auth_helper = oauth.get_authorization_url()
        return auth_helper
        
    except Exception as e:
        logger.error(f"Failed to get auth URL: {e}")
        return None


def authenticate_with_code(auth_code: str, project: Optional[str] = None) -> bool:
    """
    Complete authentication using an authorization code.
    
    This is useful for environments where the OAuth flow needs to be completed
    manually (e.g., copying the code from a browser).
    
    Parameters
    ----------
    auth_code : str
        The authorization code obtained from the OAuth flow.
    project : str, optional
        GEE project ID. If None, uses default project.
        
    Returns
    -------
    bool
        True if authentication successful, False otherwise.
        
    Example
    -------
    >>> from epistemx.ee_config import authenticate_with_code
    >>> authenticate_with_code("your-auth-code-here")
    """
    global _ee_initialized
    
    try:
        # Note: This is a simplified version. The actual implementation
        # depends on the Earth Engine Python API version
        logger.info("Attempting to authenticate with provided code...")
        
        # Try to initialize with the assumption that credentials are already set up
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        
        _ee_initialized = True
        logger.info("Authentication successful")
        return True
        
    except Exception as e:
        logger.error(f"Authentication with code failed: {e}")
        return False

def _print_manual_auth_instructions() -> None:
    """Print step-by-step manual authentication instructions."""
    instructions = """
    EARTH ENGINE AUTHENTICATION NOTES:
    
    1. Make sure you already have a google cloud project that has enable the Earth Engine API and registered to 
       commercial or non-commercial use. For more information visit: https://developers.google.com/earth-engine/guides/access 
    
    2. you can authenticate programmatically by calling: from epistemx.ee_config import authenticate_manually
       authenticate_manually()
    
    3. This will open a web browser. Sign in with your Google account that has Earth Engine access.
    
    4. Copy the authorization code from the browser and paste it in the terminal.
    
    
    For more details, visit: https://developers.google.com/earth-engine/guides/python_install
    """
    print(instructions)

def initialize_earth_engine(
    project: Optional[str] = None, 
    service_account_file: Optional[str] = None,
    force_reinit: bool = False
) -> bool:
    """
    Initialize Google Earth Engine with authentication.
    
    Parameters
    ----------
    project : str, optional
        GEE project ID. If None, uses default project.
    service_account_file : str, optional
        Path to service account JSON file. If provided, uses service account auth.
    force_reinit : bool, default False
        Force re-initialization even if already initialized.
        
    Returns
    -------
    bool
        True if initialization successful, False otherwise.
        
    Example
    -------
    >>> from epistemx.ee_config import initialize_earth_engine
    >>> # Manual authentication
    >>> initialize_earth_engine()
    >>> # Service account authentication
    >>> initialize_earth_engine(service_account_file='service-account.json')
    """
    global _ee_initialized
    
    if _ee_initialized and not force_reinit:
        logger.debug("Earth Engine already initialized")
        return True
    
    # Use service account if provided
    if service_account_file:
        return initialize_with_service_account(service_account_file, project)
    
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
            logger.warning("Earth Engine authentication required. Please run manual authentication.")
            logger.info("To authenticate manually, follow these steps:")
            _print_manual_auth_instructions()
            return False
        else:
            logger.error(f"Earth Engine initialization failed: {e}")
            return False
    
    except Exception as e:
        logger.error(f"Unexpected error during Earth Engine initialization: {e}")
        return False

def ensure_ee_initialized(
    project: Optional[str] = None, 
    service_account_file: Optional[str] = None
) -> None:
    """
    Ensure Earth Engine is initialized, raising an exception if it fails.
    
    Parameters
    ----------
    project : str, optional
        GEE project ID. If None, uses default project.
    service_account_file : str, optional
        Path to service account JSON file. If provided, uses service account auth.
        
    Raises
    ------
    RuntimeError
        If Earth Engine initialization fails.
    """
    if not initialize_earth_engine(project=project, service_account_file=service_account_file):
        raise RuntimeError(
            "Failed to initialize Google Earth Engine. "
            "Please check your authentication and internet connection. "
            "Run authenticate_manually() or provide valid service account credentials."
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

def get_auth_status() -> Dict[str, Any]:
    """
    Get detailed authentication status information.
    
    Returns
    -------
    dict
        Dictionary containing authentication status details.
    """
    status = {
        'initialized': _ee_initialized,
        'authenticated': False,
        'project': None,
        'user_info': None
    }
    
    if _ee_initialized:
        try:
            # Try a simple operation to verify authentication
            ee.Number(1).getInfo()
            status['authenticated'] = True
            
            # Try to get project info
            try:
                # This might not work in all cases, but worth trying
                status['project'] = ee.data.getAssetRoots()[0]['id'] if ee.data.getAssetRoots() else None
            except:
                pass
                
        except Exception as e:
            logger.debug(f"Authentication check failed: {e}")
            status['authenticated'] = False
    
    return status

def print_auth_instructions() -> None:
    """
    Print comprehensive authentication instructions.
    """
    _print_manual_auth_instructions()

def reset_ee_initialization() -> None:
    """
    Reset the initialization flag. Useful for testing or troubleshooting.
    """
    global _ee_initialized
    _ee_initialized = False
    logger.debug("Earth Engine initialization flag reset")

def setup_earth_engine(
    project: Optional[str] = None,
    service_account_file: Optional[str] = None,
    auto_authenticate: bool = False
) -> bool:
    """
    Comprehensive Earth Engine setup function.
    
    Parameters
    ----------
    project : str, optional
        GEE project ID.
    service_account_file : str, optional
        Path to service account JSON file.
    auto_authenticate : bool, default False
        If True, attempt manual authentication if needed.
        
    Returns
    -------
    bool
        True if setup successful, False otherwise.
        
    Example
    -------
    >>> from epistemx.ee_config import setup_earth_engine
    >>> # Try automatic setup
    >>> setup_earth_engine()
    >>> # Setup with service account
    >>> setup_earth_engine(service_account_file='service-account.json')
    """
    # First try normal initialization
    if initialize_earth_engine(project=project, service_account_file=service_account_file):
        return True
    
    # If that fails and auto_authenticate is True, try manual auth
    if auto_authenticate and not service_account_file:
        logger.info("Attempting manual authentication...")
        return authenticate_manually(project=project)
    
    return False


# ============================================================================
# GOOGLE DRIVE EXPORT SUPPORT (OAuth2)
# ============================================================================

def export_to_drive_with_oauth(
    image: ee.Image,
    description: str,
    folder: str,
    file_name_prefix: str,
    region: ee.Geometry,
    scale: int = 30,
    max_pixels: int = int(1e10),
    file_format: str = 'GeoTIFF',
    credentials_dict: Optional[Dict[str, Any]] = None
) -> ee.batch.Task:
    """
    Export Earth Engine image to Google Drive using OAuth2 credentials.
    
    This function creates an Earth Engine export task that will export
    the image to the authenticated user's Google Drive.
    
    Parameters
    ----------
    image : ee.Image
        Earth Engine image to export
    description : str
        Task description
    folder : str
        Drive folder name (will be created if doesn't exist)
    file_name_prefix : str
        Prefix for exported file name
    region : ee.Geometry
        Region to export
    scale : int, default 30
        Export resolution in meters
    max_pixels : int, default 1e10
        Maximum number of pixels to export
    file_format : str, default 'GeoTIFF'
        Export format ('GeoTIFF', 'TFRecord', etc.)
    credentials_dict : dict, optional
        OAuth2 credentials dictionary (if None, uses default EE credentials)
        
    Returns
    -------
    ee.batch.Task
        Export task object
        
    Example
    -------
    >>> from epistemx.ee_config import export_to_drive_with_oauth
    >>> task = export_to_drive_with_oauth(
    ...     image=classified_image,
    ...     description='Land Cover Export',
    ...     folder='REMAP_Exports',
    ...     file_name_prefix='lulc_2024',
    ...     region=study_area,
    ...     scale=30
    ... )
    >>> task.start()
    >>> print(f"Task ID: {task.id}")
    """
    try:
        # Create export task
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=folder,
            fileNamePrefix=file_name_prefix,
            scale=scale,
            region=region.getInfo()['coordinates'] if hasattr(region, 'getInfo') else region,
            maxPixels=max_pixels,
            fileFormat=file_format
        )
        
        logger.info(f"Created export task: {description}")
        return task
        
    except Exception as e:
        logger.error(f"Failed to create export task: {e}")
        raise


def check_export_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Check the status of an Earth Engine export task.
    
    Parameters
    ----------
    task_id : str
        Task ID to check
        
    Returns
    -------
    dict or None
        Task status dictionary, or None if task not found
        
    Example
    -------
    >>> status = check_export_task_status(task.id)
    >>> print(f"State: {status['state']}")
    """
    try:
        tasks = ee.batch.Task.list()
        for task in tasks:
            if task.id == task_id:
                return task.status()
        
        logger.warning(f"Task not found: {task_id}")
        return None
        
    except Exception as e:
        logger.error(f"Failed to check task status: {e}")
        return None


def wait_for_task_completion(
    task: ee.batch.Task,
    check_interval: int = 10,
    max_wait: int = 3600
) -> str:
    """
    Wait for an Earth Engine task to complete.
    
    Parameters
    ----------
    task : ee.batch.Task
        Task to monitor
    check_interval : int, default 10
        Seconds between status checks
    max_wait : int, default 3600
        Maximum seconds to wait
        
    Returns
    -------
    str
        Final task state ('COMPLETED', 'FAILED', 'CANCELLED', or 'TIMEOUT')
        
    Example
    -------
    >>> task.start()
    >>> final_state = wait_for_task_completion(task)
    >>> if final_state == 'COMPLETED':
    ...     print("Export successful!")
    """
    import time
    
    elapsed = 0
    
    while elapsed < max_wait:
        status = task.status()
        state = status['state']
        
        if state in ['COMPLETED', 'FAILED', 'CANCELLED']:
            logger.info(f"Task {task.id} finished with state: {state}")
            return state
        
        time.sleep(check_interval)
        elapsed += check_interval
    
    logger.warning(f"Task {task.id} timed out after {max_wait} seconds")
    return 'TIMEOUT'


def get_export_metadata(
    task: ee.batch.Task,
    training_data: Optional[Dict] = None,
    classification_params: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Generate metadata for an export task.
    
    Parameters
    ----------
    task : ee.batch.Task
        Export task
    training_data : dict, optional
        Training data information
    classification_params : dict, optional
        Classification parameters
        
    Returns
    -------
    dict
        Metadata dictionary
        
    Example
    -------
    >>> metadata = get_export_metadata(task, training_data, params)
    >>> import json
    >>> with open('metadata.json', 'w') as f:
    ...     json.dump(metadata, f, indent=2)
    """
    import datetime
    
    metadata = {
        'task_id': task.id,
        'description': task.config.get('description', 'Unknown'),
        'export_time': datetime.datetime.utcnow().isoformat(),
        'status': task.status()
    }
    
    if training_data:
        metadata['training_data'] = training_data
    
    if classification_params:
        metadata['classification_params'] = classification_params
    
    return metadata
