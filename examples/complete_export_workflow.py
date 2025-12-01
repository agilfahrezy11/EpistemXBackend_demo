"""
Complete Export Workflow Example

This example demonstrates the full workflow from Earth Engine authentication
to exporting results to Google Drive using both service account and OAuth2.
"""

import streamlit as st
import ee
from epistemx import (
    # Earth Engine authentication
    is_ee_initialized,
    initialize_with_service_account,
    get_auth_status,
    
    # Drive authentication
    DriveAuthManager,
    DriveHelper,
    ensure_valid_credentials,
    
    # Export functions
    export_to_drive_with_oauth,
    check_export_task_status,
    wait_for_task_completion,
    get_export_metadata
)

# ============================================================================
# STEP 1: EARTH ENGINE AUTHENTICATION (Service Account)
# ============================================================================

def setup_earth_engine():
    """Initialize Earth Engine with service account."""
    if not is_ee_initialized():
        st.warning("Earth Engine not initialized. Attempting to initialize...")
        
        # Try to initialize with service account
        service_account_file = 'auth/service-account.json'
        success = initialize_with_service_account(service_account_file)
        
        if success:
            st.success("✅ Earth Engine initialized successfully!")
            return True
        else:
            st.error("❌ Failed to initialize Earth Engine")
            st.info("Please go to the Earth Engine Authentication page to set up credentials.")
            return False
    
    return True


# ============================================================================
# STEP 2: DRIVE AUTHENTICATION (OAuth2)
# ============================================================================

def check_drive_authentication():
    """Check if user is authenticated with Google Drive."""
    if 'drive_credentials' not in st.session_state:
        return False
    
    # Ensure credentials are still valid
    updated_creds = ensure_valid_credentials(st.session_state['drive_credentials'])
    if updated_creds:
        st.session_state['drive_credentials'] = updated_creds
        return True
    else:
        st.error("Drive credentials expired. Please re-authenticate.")
        return False


# ============================================================================
# STEP 3: CREATE EARTH ENGINE IMAGE
# ============================================================================

def create_sample_classification():
    """Create a sample classified image for demonstration."""
    
    # Define region of interest
    region = ee.Geometry.Rectangle([-122.5, 37.5, -122.0, 38.0])
    
    # Load Landsat 8 imagery
    landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(region) \
        .filterDate('2020-01-01', '2020-12-31') \
        .median()
    
    # Select bands
    bands = ['SR_B4', 'SR_B3', 'SR_B2']  # RGB
    image = landsat.select(bands)
    
    # Calculate NDVI
    nir = landsat.select('SR_B5')
    red = landsat.select('SR_B4')
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    
    # Simple classification based on NDVI
    # Water: NDVI < 0
    # Bare soil: 0 <= NDVI < 0.2
    # Vegetation: NDVI >= 0.2
    classified = ee.Image(1) \
        .where(ndvi.lt(0), 0) \
        .where(ndvi.gte(0).And(ndvi.lt(0.2)), 1) \
        .where(ndvi.gte(0.2), 2) \
        .rename('classification')
    
    return classified, region


# ============================================================================
# STEP 4: EXPORT TO DRIVE
# ============================================================================

def export_classification_to_drive(image, region, description):
    """Export classified image to Google Drive."""
    
    try:
        # Create export task
        task = export_to_drive_with_oauth(
            image=image,
            description=description,
            folder='EpistemX_Exports',
            file_name_prefix=description.replace(' ', '_'),
            region=region,
            scale=30,
            max_pixels=int(1e10),
            file_format='GeoTIFF'
        )
        
        # Start the task
        task.start()
        
        st.success(f"✅ Export task started!")
        st.info(f"Task ID: {task.id}")
        
        # Store task ID in session
        st.session_state['export_task_id'] = task.id
        st.session_state['export_description'] = description
        
        return task
        
    except Exception as e:
        st.error(f"Failed to start export: {str(e)}")
        return None


# ============================================================================
# STEP 5: MONITOR EXPORT STATUS
# ============================================================================

def monitor_export_status():
    """Monitor the status of an ongoing export."""
    
    if 'export_task_id' not in st.session_state:
        st.info("No active export task.")
        return
    
    task_id = st.session_state['export_task_id']
    description = st.session_state.get('export_description', 'Unknown')
    
    st.subheader(f"Export Status: {description}")
    
    # Check status
    status = check_export_task_status(task_id)
    
    if status:
        state = status['state']
        
        # Display status with appropriate styling
        if state == 'COMPLETED':
            st.success("✅ Export completed successfully!")
            st.balloons()
            
            # Provide link to Drive
            st.markdown("""
            Your export is now available in your Google Drive:
            - Folder: **EpistemX_Exports**
            - [Open Google Drive](https://drive.google.com/drive/)
            """)
            
            # Clear task from session
            if st.button("Clear Export Status"):
                del st.session_state['export_task_id']
                del st.session_state['export_description']
                st.rerun()
                
        elif state == 'RUNNING':
            st.info("⏳ Export in progress...")
            st.progress(0.5)  # Indeterminate progress
            
            # Auto-refresh button
            if st.button("🔄 Refresh Status"):
                st.rerun()
                
        elif state == 'FAILED':
            st.error("❌ Export failed!")
            error_message = status.get('error_message', 'Unknown error')
            st.error(f"Error: {error_message}")
            
            # Clear task from session
            if st.button("Clear Export Status"):
                del st.session_state['export_task_id']
                del st.session_state['export_description']
                st.rerun()
                
        else:
            st.warning(f"Status: {state}")
    else:
        st.error("Could not retrieve task status.")


# ============================================================================
# STEP 6: CREATE METADATA FILE
# ============================================================================

def create_and_upload_metadata(task, training_data=None, params=None):
    """Create metadata file and upload to Drive."""
    
    if not check_drive_authentication():
        st.error("Drive authentication required to upload metadata.")
        return
    
    try:
        # Generate metadata
        metadata = get_export_metadata(task, training_data, params)
        
        # Convert to JSON string
        import json
        metadata_json = json.dumps(metadata, indent=2)
        
        # Get Drive credentials
        credentials = DriveAuthManager.dict_to_credentials(
            st.session_state['drive_credentials']
        )
        
        # Create Drive helper
        drive_helper = DriveHelper(credentials)
        
        # Find or create export folder
        folder_id = drive_helper.create_folder('EpistemX_Exports')
        
        # Upload metadata file
        file_id = drive_helper.upload_file(
            f"{st.session_state['export_description']}_metadata.json",
            metadata_json.encode('utf-8'),
            'application/json',
            folder_id
        )
        
        st.success(f"✅ Metadata uploaded! File ID: {file_id}")
        
    except Exception as e:
        st.error(f"Failed to upload metadata: {str(e)}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(
        page_title="Complete Export Workflow",
        page_icon="🌍",
        layout="wide"
    )
    
    st.title("🌍 Complete Export Workflow Example")
    st.markdown("""
    This example demonstrates the complete workflow from Earth Engine authentication
    to exporting results to Google Drive.
    """)
    
    # Check Earth Engine authentication
    st.header("1️⃣ Earth Engine Authentication")
    if setup_earth_engine():
        status = get_auth_status()
        st.success(f"✅ Earth Engine authenticated")
        if status.get('project'):
            st.info(f"Project: {status['project']}")
    else:
        st.stop()
    
    st.divider()
    
    # Check Drive authentication
    st.header("2️⃣ Google Drive Authentication")
    if check_drive_authentication():
        st.success(f"✅ Authenticated as: {st.session_state['drive_user_email']}")
    else:
        st.warning("⚠️ Not authenticated with Google Drive")
        st.info("👉 Go to the **Google Drive Export Setup** page to authenticate.")
        st.stop()
    
    st.divider()
    
    # Create and export classification
    st.header("3️⃣ Create and Export Classification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classification Parameters")
        
        description = st.text_input(
            "Export Description",
            value="Land Cover Classification",
            help="Description for the export task"
        )
        
        scale = st.number_input(
            "Export Scale (meters)",
            min_value=10,
            max_value=1000,
            value=30,
            help="Resolution of the exported image"
        )
        
        if st.button("🚀 Create and Export Classification", type="primary"):
            with st.spinner("Creating classification..."):
                # Create classification
                classified, region = create_sample_classification()
                
                st.success("✅ Classification created!")
                
                # Export to Drive
                task = export_classification_to_drive(classified, region, description)
                
                if task:
                    st.rerun()
    
    with col2:
        st.subheader("Classification Info")
        st.markdown("""
        **Classes:**
        - 0: Water (NDVI < 0)
        - 1: Bare Soil (0 ≤ NDVI < 0.2)
        - 2: Vegetation (NDVI ≥ 0.2)
        
        **Region:**
        - San Francisco Bay Area
        - Coordinates: [-122.5, 37.5, -122.0, 38.0]
        
        **Data Source:**
        - Landsat 8 Collection 2 Level 2
        - Year: 2020
        """)
    
    st.divider()
    
    # Monitor export status
    st.header("4️⃣ Export Status")
    monitor_export_status()
    
    st.divider()
    
    # Additional options
    st.header("5️⃣ Additional Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Metadata")
        if st.button("📄 Create and Upload Metadata"):
            if 'export_task_id' in st.session_state:
                # Get task
                tasks = ee.batch.Task.list()
                task = next((t for t in tasks if t.id == st.session_state['export_task_id']), None)
                
                if task:
                    create_and_upload_metadata(task)
                else:
                    st.error("Task not found")
            else:
                st.warning("No active export task")
    
    with col2:
        st.subheader("Test Drive Access")
        if st.button("🧪 Create Test Folder"):
            if check_drive_authentication():
                try:
                    credentials = DriveAuthManager.dict_to_credentials(
                        st.session_state['drive_credentials']
                    )
                    drive_helper = DriveHelper(credentials)
                    folder_id = drive_helper.create_folder('EpistemX_Test')
                    st.success(f"✅ Test folder created! ID: {folder_id}")
                    st.markdown(f"[View in Drive](https://drive.google.com/drive/folders/{folder_id})")
                except Exception as e:
                    st.error(f"Failed: {str(e)}")


if __name__ == "__main__":
    main()
