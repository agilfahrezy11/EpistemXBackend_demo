"""
Google Drive Export Setup Module

This module provides OAuth2 authentication for exporting Earth Engine
results to the user's personal Google Drive.
"""
import streamlit as st
import os
import json
import tempfile
from epistemx import (
    DriveAuthManager,
    DriveHelper,
    ensure_valid_credentials,
    is_ee_initialized
)

# Page configuration
st.set_page_config(
    page_title="Google Drive Export Setup",
    page_icon="☁️",
    layout="wide"
)

# Load custom CSS
def load_css():
    """Load custom CSS for EpistemX theme"""
    try:
        with open('.streamlit/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css()

# Page header
st.markdown("""
<div class="main-header">
    <h1>☁️ Google Drive Export Setup</h1>
    <p>Configure OAuth2 authentication for exporting results to your Google Drive</p>
</div>
""", unsafe_allow_html=True)

# Check Earth Engine authentication first
if not is_ee_initialized():
    st.warning("⚠️ Earth Engine is not authenticated. Please authenticate Earth Engine first before setting up Drive exports.")
    st.info("👉 Go to the **Earth Engine Authentication** page to authenticate.")
    st.stop()

# Initialize session state
if 'drive_credentials' not in st.session_state:
    st.session_state.drive_credentials = None
if 'drive_user_email' not in st.session_state:
    st.session_state.drive_user_email = None
if 'oauth_state' not in st.session_state:
    st.session_state.oauth_state = None

# Display current Drive authentication status
st.markdown('<div class="module-header">📊 Drive Authentication Status</div>', unsafe_allow_html=True)

if st.session_state.drive_credentials:
    # Check if credentials are still valid
    updated_creds = ensure_valid_credentials(st.session_state.drive_credentials)
    if updated_creds:
        st.session_state.drive_credentials = updated_creds
        st.success(f"✅ Authenticated as: {st.session_state.drive_user_email}")
        st.info("You can now export Earth Engine results to your Google Drive!")
        
        if st.button("🔄 Sign Out from Drive", type="secondary"):
            st.session_state.drive_credentials = None
            st.session_state.drive_user_email = None
            st.session_state.oauth_state = None
            st.rerun()
    else:
        st.error("❌ Drive credentials expired and could not be refreshed. Please sign in again.")
        st.session_state.drive_credentials = None
        st.session_state.drive_user_email = None
else:
    st.warning("⚠️ Not authenticated with Google Drive. Please authenticate below to enable exports.")

st.divider()

# OAuth2 Setup Section
st.markdown('<div class="module-header">🔐 OAuth2 Authentication Setup</div>', unsafe_allow_html=True)

# Check for OAuth client secrets file
oauth_secrets_file = None
possible_locations = [
    'secrets/oauth_client_secret.json',
    'auth/oauth_client_secret.json',
    'oauth_client_secret.json',
    '.streamlit/secrets/oauth_client_secret.json'
]

for location in possible_locations:
    if os.path.exists(location):
        oauth_secrets_file = location
        break

if not oauth_secrets_file:
    st.error("❌ OAuth2 client secrets file not found!")
    
    with st.expander("📖 How to Create OAuth2 Credentials", expanded=True):
        st.markdown("""
        ### Creating OAuth2 Credentials for Drive Export
        
        #### Step 1: Go to Google Cloud Console
        1. Visit [Google Cloud Console](https://console.cloud.google.com/)
        2. Select your project (or create a new one)
        
        #### Step 2: Enable Google Drive API
        1. Go to "APIs & Services" > "Library"
        2. Search for "Google Drive API"
        3. Click "Enable"
        
        #### Step 3: Create OAuth2 Credentials
        1. Go to "APIs & Services" > "Credentials"
        2. Click "Create Credentials" > "OAuth client ID"
        3. Choose "Web application"
        4. Add authorized redirect URIs:
           - For local development: `http://localhost:8501`
           - For Streamlit Cloud: `https://your-app-name.streamlit.app`
        5. Click "Create"
        
        #### Step 4: Download Credentials
        1. Click the download button (⬇️) next to your OAuth client
        2. Save the file as `oauth_client_secret.json`
        3. Place it in one of these locations:
           - `secrets/oauth_client_secret.json` (recommended)
           - `auth/oauth_client_secret.json`
           - Project root directory
        
        #### Step 5: File Format
        Your `oauth_client_secret.json` should look like:
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
        
        #### For Streamlit Cloud Deployment
        Use Streamlit secrets instead of a file:
        1. Go to your app settings on Streamlit Cloud
        2. Add to secrets.toml:
        ```toml
        [oauth_client]
        client_id = "YOUR_CLIENT_ID.apps.googleusercontent.com"
        client_secret = "YOUR_CLIENT_SECRET"
        redirect_uri = "https://your-app-name.streamlit.app"
        ```
        """)
    
    # Allow manual input of OAuth credentials
    st.markdown("### 🔧 Manual OAuth Configuration")
    
    with st.form("manual_oauth_config"):
        st.info("If you can't upload a file, you can manually enter your OAuth credentials here.")
        
        client_id = st.text_input(
            "Client ID:",
            placeholder="YOUR_CLIENT_ID.apps.googleusercontent.com",
            help="Your OAuth2 client ID from Google Cloud Console"
        )
        
        client_secret = st.text_input(
            "Client Secret:",
            type="password",
            placeholder="YOUR_CLIENT_SECRET",
            help="Your OAuth2 client secret"
        )
        
        redirect_uri = st.text_input(
            "Redirect URI:",
            value="http://localhost:8501",
            help="Must match the redirect URI configured in Google Cloud Console"
        )
        
        if st.form_submit_button("💾 Save OAuth Configuration", type="primary"):
            if client_id and client_secret:
                # Create OAuth secrets structure
                oauth_config = {
                    "web": {
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [redirect_uri]
                    }
                }
                
                # Save to secrets directory
                os.makedirs('secrets', exist_ok=True)
                oauth_secrets_file = 'secrets/oauth_client_secret.json'
                
                with open(oauth_secrets_file, 'w') as f:
                    json.dump(oauth_config, f, indent=2)
                
                st.success(f"✅ OAuth configuration saved to {oauth_secrets_file}")
                st.rerun()
            else:
                st.error("❌ Please provide both Client ID and Client Secret")
    
    st.stop()

# OAuth2 Authentication Flow
st.markdown("### 🔑 Authenticate with Google Drive")

# Check if we're handling an OAuth callback
query_params = st.query_params

if 'code' in query_params and not st.session_state.drive_credentials:
    auth_code = query_params['code']
    
    with st.spinner("Completing authentication..."):
        try:
            # Get redirect URI from secrets file
            with open(oauth_secrets_file, 'r') as f:
                oauth_config = json.load(f)
            
            redirect_uri = oauth_config['web']['redirect_uris'][0]
            
            # Exchange code for credentials
            auth_manager = DriveAuthManager(oauth_secrets_file, redirect_uri)
            credentials, user_info = auth_manager.exchange_code_for_credentials(auth_code)
            
            if credentials and user_info:
                # Store credentials in session
                st.session_state.drive_credentials = DriveAuthManager.credentials_to_dict(credentials)
                st.session_state.drive_user_email = user_info.get('email')
                
                # Clear query params
                st.query_params.clear()
                
                st.success(f"✅ Successfully authenticated as {user_info.get('email')}!")
                st.balloons()
                st.rerun()
            else:
                st.error("❌ Authentication failed. Please try again.")
                
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")

# Show authentication button if not authenticated
if not st.session_state.drive_credentials:
    st.markdown("""
    Click the button below to authenticate with your Google account.
    You will be redirected to Google's authorization page.
    """)
    
    try:
        # Get redirect URI from secrets file
        with open(oauth_secrets_file, 'r') as f:
            oauth_config = json.load(f)
        
        redirect_uri = oauth_config['web']['redirect_uris'][0]
        
        # Create auth manager
        auth_manager = DriveAuthManager(oauth_secrets_file, redirect_uri)
        
        if st.button("🔗 Sign in with Google", type="primary", use_container_width=True):
            # Generate authorization URL
            auth_url, state = auth_manager.get_authorization_url()
            st.session_state.oauth_state = state
            
            # Display authorization link
            st.markdown(f"""
            ### 🔗 Authorization Required
            
            Click the link below to authorize this application to access your Google Drive:
            
            **[Authorize with Google]({auth_url})**
            
            After authorizing, you will be redirected back to this page.
            """)
            
            st.info("""
            **What permissions are requested?**
            - Access to create files in your Google Drive
            - Access to your email address (for identification)
            
            **Note:** This application will only be able to access files it creates.
            It cannot access your existing Drive files.
            """)
    
    except Exception as e:
        st.error(f"Failed to initialize OAuth: {str(e)}")

# Test Drive Access Section
if st.session_state.drive_credentials:
    st.divider()
    st.markdown('<div class="module-header">🧪 Test Drive Access</div>', unsafe_allow_html=True)
    
    st.markdown("Test your Drive authentication by creating a test folder.")
    
    test_folder_name = st.text_input(
        "Test Folder Name:",
        value="EpistemX_Test_Folder",
        help="Name for the test folder to create in your Drive"
    )
    
    if st.button("🧪 Create Test Folder", type="secondary"):
        with st.spinner("Creating test folder..."):
            try:
                # Get credentials
                credentials = DriveAuthManager.dict_to_credentials(st.session_state.drive_credentials)
                
                # Create Drive helper
                drive_helper = DriveHelper(credentials)
                
                # Create test folder
                folder_id = drive_helper.create_folder(test_folder_name)
                
                st.success(f"✅ Test folder created successfully!")
                st.info(f"Folder ID: {folder_id}")
                st.markdown(f"[View in Google Drive](https://drive.google.com/drive/folders/{folder_id})")
                
            except Exception as e:
                st.error(f"Failed to create test folder: {str(e)}")

# Information Section
st.divider()
st.markdown("""
<div class="epistemx-card">
    <h4>📚 About Drive Export Authentication</h4>
    <p>
    This authentication allows the application to export Earth Engine processing results
    directly to your personal Google Drive. The OAuth2 flow ensures that:
    </p>
    <ul>
        <li>You maintain full control over your Drive access</li>
        <li>The application can only access files it creates</li>
        <li>You can revoke access at any time from your Google Account settings</li>
        <li>Your credentials are stored securely in your browser session</li>
    </ul>
    
    <h4>🔒 Security & Privacy</h4>
    <ul>
        <li><strong>Credentials Storage:</strong> OAuth tokens are stored only in your browser session</li>
        <li><strong>Limited Scope:</strong> Only Drive file creation and email access are requested</li>
        <li><strong>Automatic Refresh:</strong> Expired tokens are automatically refreshed</li>
        <li><strong>Revocation:</strong> You can revoke access anytime at <a href="https://myaccount.google.com/permissions" target="_blank">Google Account Permissions</a></li>
    </ul>
    
    <h4>🔗 Helpful Resources</h4>
    <ul>
        <li><a href="https://console.cloud.google.com/" target="_blank">Google Cloud Console</a></li>
        <li><a href="https://developers.google.com/drive/api/guides/about-auth" target="_blank">Drive API Authentication Guide</a></li>
        <li><a href="https://myaccount.google.com/permissions" target="_blank">Manage App Permissions</a></li>
    </ul>
</div>
""", unsafe_allow_html=True)
