# Google Drive Export Authentication Guide

## Overview

This guide explains how to set up and use Google Drive export functionality in EpistemX, which allows you to export Earth Engine processing results directly to your personal Google Drive.

## Two Authentication Systems

EpistemX uses **two separate authentication mechanisms**:

### 1. Service Account (Earth Engine API Access)
- **Purpose:** Access Google Earth Engine for satellite imagery processing
- **Type:** Application-level credentials
- **Setup:** One-time configuration with service account JSON key
- **User Interaction:** None required after setup
- **Used For:** 
  - Loading satellite imagery
  - Running classifications
  - Processing Earth Engine operations

### 2. OAuth2 (Google Drive Access)
- **Purpose:** Export results to user's personal Google Drive
- **Type:** User-specific credentials
- **Setup:** User must authorize the application
- **User Interaction:** Required for each user
- **Used For:**
  - Creating folders in user's Drive
  - Uploading export files (GeoTIFF, CSV, etc.)
  - Managing export results

## Why Two Systems?

The separation ensures:
- **Earth Engine operations** run with reliable service account credentials
- **Drive exports** go to each user's personal Drive (not a shared account)
- **User privacy** is maintained (each user controls their own exports)
- **Scalability** for multiple users without credential conflicts

---

## Part 1: Service Account Setup (Earth Engine)

### Prerequisites
- Google Cloud Project with Earth Engine API enabled
- Service account created with Earth Engine permissions
- Service account JSON key file downloaded

### Setup Steps

1. **Create Service Account** (if not already done)
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Navigate to "IAM & Admin" > "Service Accounts"
   - Create service account with Earth Engine permissions
   - Download JSON key file

2. **Place Service Account File**
   - Save as `auth/service-account.json` in your project
   - Or set environment variable: `GOOGLE_APPLICATION_CREDENTIALS`

3. **Authenticate in EpistemX**
   - Go to "Earth Engine Authentication" page
   - Upload or paste service account JSON
   - Click "Authenticate"

### Verification
```python
from epistemx import is_ee_initialized, get_auth_status

# Check if Earth Engine is ready
if is_ee_initialized():
    print("✅ Earth Engine authenticated")
    status = get_auth_status()
    print(f"Project: {status.get('project')}")
```

---

## Part 2: OAuth2 Setup (Google Drive)

### Prerequisites
- Google Cloud Project (same as Earth Engine)
- Google Drive API enabled
- OAuth2 client credentials created

### Step 1: Enable Google Drive API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project
3. Navigate to "APIs & Services" > "Library"
4. Search for "Google Drive API"
5. Click "Enable"

### Step 2: Create OAuth2 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. Choose "Web application"
4. Configure:
   - **Name:** EpistemX Drive Export
   - **Authorized redirect URIs:**
     - For local: `http://localhost:8501`
     - For Streamlit Cloud: `https://your-app-name.streamlit.app`
5. Click "Create"
6. Download the JSON file

### Step 3: Configure OAuth in EpistemX

#### Option A: Upload File (Recommended)

1. Save downloaded file as `secrets/oauth_client_secret.json`
2. File structure should be:
```json
{
  "web": {
    "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
    "client_secret": "YOUR_CLIENT_SECRET",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "redirect_uris": ["http://localhost:8501"]
  }
}
```

#### Option B: Manual Entry

1. Go to "Google Drive Export Setup" page
2. Click "Manual OAuth Configuration"
3. Enter Client ID and Client Secret
4. Click "Save OAuth Configuration"

#### Option C: Streamlit Secrets (For Cloud Deployment)

Add to `.streamlit/secrets.toml`:
```toml
[oauth_client]
client_id = "YOUR_CLIENT_ID.apps.googleusercontent.com"
client_secret = "YOUR_CLIENT_SECRET"
redirect_uri = "https://your-app-name.streamlit.app"
```

### Step 4: Authenticate with Google

1. Go to "Google Drive Export Setup" page
2. Click "Sign in with Google"
3. You'll be redirected to Google's authorization page
4. Review permissions:
   - Create files in your Google Drive
   - Access your email address
5. Click "Allow"
6. You'll be redirected back to EpistemX

### Step 5: Verify Authentication

The page will show:
- ✅ Authenticated as: your-email@gmail.com
- You can now export Earth Engine results to your Google Drive!

---

## Using Drive Export

### In Python Code

```python
import ee
from epistemx import (
    export_to_drive_with_oauth,
    check_export_task_status,
    wait_for_task_completion
)

# Create an Earth Engine image (example)
image = ee.Image('USGS/SRTMGL1_003')
region = ee.Geometry.Rectangle([-122.5, 37.5, -122.0, 38.0])

# Export to Drive
task = export_to_drive_with_oauth(
    image=image,
    description='Elevation Export',
    folder='EpistemX_Exports',
    file_name_prefix='elevation_map',
    region=region,
    scale=30
)

# Start the export
task.start()
print(f"Export started. Task ID: {task.id}")

# Wait for completion (optional)
final_state = wait_for_task_completion(task)
if final_state == 'COMPLETED':
    print("✅ Export completed! Check your Google Drive.")
```

### In Streamlit App

```python
import streamlit as st
from epistemx import DriveAuthManager, DriveHelper

# Check if user is authenticated
if 'drive_credentials' in st.session_state:
    # Get credentials
    credentials = DriveAuthManager.dict_to_credentials(
        st.session_state['drive_credentials']
    )
    
    # Create Drive helper
    drive_helper = DriveHelper(credentials)
    
    # Create export folder
    folder_id = drive_helper.create_folder('My Exports')
    
    # Upload a file
    file_content = b"Sample data"
    file_id = drive_helper.upload_file(
        'results.csv',
        file_content,
        'text/csv',
        folder_id
    )
    
    st.success(f"File uploaded! ID: {file_id}")
else:
    st.warning("Please authenticate with Google Drive first")
```

---

## OAuth2 Flow Explained

### The Complete Flow

```
1. User clicks "Sign in with Google"
   │
2. Application generates authorization URL
   │
3. User is redirected to Google's consent page
   │
4. User reviews permissions and clicks "Allow"
   │
5. Google redirects back with authorization code
   │
6. Application exchanges code for credentials
   │  ├─ Access token (expires in 1 hour)
   │  └─ Refresh token (long-lived)
   │
7. Credentials stored in session state
   │
8. User can now export to Drive
   │
9. Access token automatically refreshed when expired
```

### What Gets Stored

In `st.session_state['drive_credentials']`:
```python
{
    'token': 'ya29.a0AfH6SMB...',           # Access token (1 hour)
    'refresh_token': '1//0gXXXXXXXXXXXX',  # Refresh token (long-lived)
    'token_uri': 'https://oauth2.googleapis.com/token',
    'client_id': 'YOUR_CLIENT_ID.apps.googleusercontent.com',
    'client_secret': 'YOUR_CLIENT_SECRET',
    'scopes': [
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/userinfo.email'
    ]
}
```

### Automatic Token Refresh

```python
from epistemx import ensure_valid_credentials

# Before any Drive operation
updated_creds = ensure_valid_credentials(
    st.session_state['drive_credentials']
)

if updated_creds:
    st.session_state['drive_credentials'] = updated_creds
    # Proceed with Drive operation
else:
    # Credentials expired and couldn't refresh
    # User needs to re-authenticate
```

---

## Security & Privacy

### What Permissions Are Requested?

1. **`drive.file` scope**
   - Allows creating files in your Drive
   - **Cannot** access existing files you didn't create with this app
   - **Cannot** delete files you created elsewhere

2. **`userinfo.email` scope**
   - Allows reading your email address
   - Used for user identification only

### Where Are Credentials Stored?

- **Session State:** Credentials stored in `st.session_state` (browser memory)
- **Not Persistent:** Cleared when you close the browser
- **Not Shared:** Each user has their own credentials
- **Server-Side:** Never sent to client-side JavaScript

### How to Revoke Access

1. Go to [Google Account Permissions](https://myaccount.google.com/permissions)
2. Find "EpistemX" (or your app name)
3. Click "Remove Access"
4. Credentials will no longer work

### Best Practices

✅ **DO:**
- Keep OAuth client secrets secure
- Use HTTPS in production
- Request minimal scopes needed
- Implement automatic token refresh
- Clear credentials on sign out

❌ **DON'T:**
- Commit OAuth secrets to version control
- Share OAuth credentials between users
- Request unnecessary permissions
- Store credentials in plain text files
- Keep expired sessions indefinitely

---

## Troubleshooting

### Issue: "OAuth2 client secrets file not found"

**Solution:**
1. Create `secrets/` directory
2. Place `oauth_client_secret.json` in it
3. Or use manual configuration option

### Issue: "Redirect URI mismatch"

**Solution:**
1. Check redirect URI in Google Cloud Console
2. Must match exactly: `http://localhost:8501` (local) or your Streamlit Cloud URL
3. Update OAuth configuration if needed

### Issue: "Access token expired"

**Solution:**
- Automatic refresh should handle this
- If it fails, sign out and sign in again
- Check that refresh token is present

### Issue: "Insufficient permissions"

**Solution:**
1. Check that Drive API is enabled
2. Verify OAuth scopes include `drive.file`
3. Re-authenticate to get updated permissions

### Issue: "Export task fails"

**Solution:**
1. Check Earth Engine authentication first
2. Verify Drive authentication is active
3. Check export parameters (region, scale, etc.)
4. Review task status: `check_export_task_status(task.id)`

---

## Example: Complete Export Workflow

```python
import streamlit as st
import ee
from epistemx import (
    is_ee_initialized,
    export_to_drive_with_oauth,
    check_export_task_status,
    DriveAuthManager,
    DriveHelper
)

# Check Earth Engine authentication
if not is_ee_initialized():
    st.error("Please authenticate Earth Engine first")
    st.stop()

# Check Drive authentication
if 'drive_credentials' not in st.session_state:
    st.error("Please authenticate Google Drive first")
    st.stop()

# Create a sample image
image = ee.Image('USGS/SRTMGL1_003')
region = ee.Geometry.Rectangle([-122.5, 37.5, -122.0, 38.0])

# Export button
if st.button("Export to Drive"):
    with st.spinner("Starting export..."):
        # Create export task
        task = export_to_drive_with_oauth(
            image=image,
            description='Elevation Export',
            folder='EpistemX_Exports',
            file_name_prefix='elevation_map',
            region=region,
            scale=30
        )
        
        # Start task
        task.start()
        
        # Store task ID in session
        st.session_state['export_task_id'] = task.id
        
        st.success(f"Export started! Task ID: {task.id}")

# Check export status
if 'export_task_id' in st.session_state:
    if st.button("Check Export Status"):
        status = check_export_task_status(st.session_state['export_task_id'])
        
        if status:
            state = status['state']
            
            if state == 'COMPLETED':
                st.success("✅ Export completed! Check your Google Drive.")
            elif state == 'RUNNING':
                st.info("⏳ Export in progress...")
            elif state == 'FAILED':
                st.error(f"❌ Export failed: {status.get('error_message')}")
            else:
                st.warning(f"Status: {state}")
```

---

## API Reference

### DriveAuthManager

```python
from epistemx import DriveAuthManager

# Initialize
auth_manager = DriveAuthManager(
    client_secrets_file='secrets/oauth_client_secret.json',
    redirect_uri='http://localhost:8501'
)

# Get authorization URL
auth_url, state = auth_manager.get_authorization_url()

# Exchange code for credentials
credentials, user_info = auth_manager.exchange_code_for_credentials(code)

# Convert credentials for storage
creds_dict = DriveAuthManager.credentials_to_dict(credentials)

# Convert back to credentials object
credentials = DriveAuthManager.dict_to_credentials(creds_dict)

# Refresh expired credentials
credentials = DriveAuthManager.refresh_credentials(credentials)
```

### DriveHelper

```python
from epistemx import DriveHelper

# Initialize with credentials
drive_helper = DriveHelper(credentials)

# Create folder
folder_id = drive_helper.create_folder('My Folder')

# Upload file
file_id = drive_helper.upload_file(
    'data.csv',
    file_content,
    'text/csv',
    folder_id
)

# Copy file
new_file_id = drive_helper.copy_file(file_id, 'data_copy.csv', folder_id)

# List files
files = drive_helper.list_files(folder_id)

# Grant access
drive_helper.grant_access(file_id, 'user@example.com', 'writer')

# Delete file
drive_helper.delete_file(file_id)
```

### Export Functions

```python
from epistemx import (
    export_to_drive_with_oauth,
    check_export_task_status,
    wait_for_task_completion,
    get_export_metadata
)

# Create export task
task = export_to_drive_with_oauth(
    image=ee_image,
    description='My Export',
    folder='Exports',
    file_name_prefix='result',
    region=ee_geometry,
    scale=30,
    max_pixels=1e10,
    file_format='GeoTIFF'
)

# Start task
task.start()

# Check status
status = check_export_task_status(task.id)

# Wait for completion
final_state = wait_for_task_completion(task, check_interval=10)

# Get metadata
metadata = get_export_metadata(task, training_data, params)
```

---

## Additional Resources

- [Google Earth Engine](https://earthengine.google.com/)
- [Google Drive API Documentation](https://developers.google.com/drive/api)
- [OAuth 2.0 Documentation](https://developers.google.com/identity/protocols/oauth2)
- [Google Cloud Console](https://console.cloud.google.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## Support

For issues or questions:
1. Check this documentation
2. Review the troubleshooting section
3. Check Earth Engine authentication first
4. Verify OAuth configuration
5. Review Google Cloud Console settings
