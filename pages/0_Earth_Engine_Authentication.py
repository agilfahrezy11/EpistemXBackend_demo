"""
Earth Engine Authentication Module

This module provides a user interface for authenticating with Google Earth Engine
using either personal account or service account credentials.
"""
import streamlit as st
import os
import json
import tempfile
from epistemx import (
    authenticate_manually,
    initialize_with_service_account,
    get_auth_status,
    reset_ee_initialization,
    is_ee_initialized
)

# Page configuration
st.set_page_config(
    page_title="Earth Engine Authentication",
    page_icon="🔐",
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
    <h1>🔐 Earth Engine Authentication</h1>
    <p>Configure your Google Earth Engine access</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'auth_method' not in st.session_state:
    st.session_state.auth_method = None
if 'auth_step' not in st.session_state:
    st.session_state.auth_step = 'select_method'

# Display current authentication status
st.markdown('<div class="module-header">📊 Current Status</div>', unsafe_allow_html=True)

auth_status = get_auth_status()
if auth_status['initialized'] and auth_status['authenticated']:
    st.success("✅ Earth Engine is authenticated and ready to use!")
    if auth_status.get('project'):
        st.info(f"**Project:** {auth_status['project']}")
    
    if st.button("🔄 Reset Authentication", type="secondary"):
        reset_ee_initialization()
        st.session_state.ee_initialized = False
        st.session_state.auth_step = 'select_method'
        st.rerun()
else:
    st.warning("⚠️ Earth Engine is not authenticated. Please authenticate below.")

st.divider()

# Authentication method selection
st.markdown('<div class="module-header">🔑 Authentication Method</div>', unsafe_allow_html=True)

auth_method = st.radio(
    "Choose your authentication method:",
    ["Personal Account (OAuth)", "Service Account (JSON Key)"],
    index=0,
    help="Personal Account: Use your Google account credentials. Service Account: Use a service account JSON key file."
)

st.divider()

# ============= PERSONAL ACCOUNT AUTHENTICATION =============
if auth_method == "Personal Account (OAuth)":
    st.markdown("""
    <div class="epistemx-card">
        <h3>🔐 Personal Account Authentication</h3>
        <p>This method uses your personal Google account to authenticate with Earth Engine.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Prerequisites")
    st.markdown("""
    1. You must have a Google account with Earth Engine access
    2. Your account must be registered for Earth Engine (commercial or non-commercial use)
    3. Visit [Google Earth Engine](https://earthengine.google.com/) to register if you haven't already
    """)
    
    st.markdown("### Authentication Steps")
    
    with st.form("personal_auth_form"):
        st.markdown("""
        **Step 1:** Click the button below to start the authentication process.
        
        **Step 2:** A browser window will open (or you'll receive a URL to visit).
        
        **Step 3:** Sign in with your Google account that has Earth Engine access.
        
        **Step 4:** Copy the authorization code from the browser.
        
        **Step 5:** Paste the authorization code in the field below.
        """)
        
        project_id = st.text_input(
            "Google Cloud Project ID (optional):",
            help="If you have a specific GCP project for Earth Engine, enter it here. Otherwise, leave blank."
        )
        
        auth_code = st.text_area(
            "Authorization Code:",
            height=100,
            placeholder="Paste your authorization code here after completing the browser authentication...",
            help="After authenticating in your browser, you'll receive a code. Paste it here."
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            submit_button = st.form_submit_button("🚀 Authenticate", type="primary")
        with col2:
            if st.form_submit_button("📖 Show Detailed Instructions"):
                st.session_state.show_instructions = True
    
    if submit_button:
        with st.spinner("Authenticating with Earth Engine..."):
            try:
                # Note: The actual OAuth flow happens outside Streamlit
                # This is a placeholder for the authentication process
                st.info("""
                **Important:** Personal account authentication requires running the authentication 
                process in your local terminal or Python environment.
                
                Please run the following in your Python environment:
                ```python
                from epistemx import authenticate_manually
                authenticate_manually()
                ```
                
                Then restart this Streamlit app.
                """)
                
                # Try to initialize if already authenticated
                if project_id:
                    success = authenticate_manually(project=project_id)
                else:
                    success = authenticate_manually()
                
                if success:
                    st.session_state.ee_initialized = True
                    st.success("✅ Authentication successful!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Authentication failed. Please try again.")
                    
            except Exception as e:
                st.error(f"Authentication error: {str(e)}")
    
    # Show detailed instructions if requested
    if st.session_state.get('show_instructions', False):
        with st.expander("📖 Detailed Authentication Instructions", expanded=True):
            st.markdown("""
            ### Complete Authentication Guide
            
            #### Option A: Authenticate in Terminal (Recommended)
            
            1. Open a terminal or command prompt
            2. Activate your Python environment (if using virtual environment)
            3. Run the following Python code:
            
            ```python
            import ee
            ee.Authenticate()
            ee.Initialize()
            ```
            
            4. Follow the prompts to authenticate in your browser
            5. Once complete, restart this Streamlit application
            
            #### Option B: Authenticate in Jupyter Notebook
            
            1. Open a Jupyter notebook
            2. Run the following code:
            
            ```python
            from epistemx import authenticate_manually
            authenticate_manually()
            ```
            
            3. Follow the authentication flow
            4. Restart this Streamlit application
            
            #### Troubleshooting
            
            - **"Not authenticated" error:** Make sure you've completed the authentication flow
            - **"No project" error:** Specify your Google Cloud Project ID
            - **"Access denied" error:** Ensure your account is registered for Earth Engine
            
            For more help, visit: [Earth Engine Python Installation Guide](https://developers.google.com/earth-engine/guides/python_install)
            """)

# ============= SERVICE ACCOUNT AUTHENTICATION =============
elif auth_method == "Service Account (JSON Key)":
    st.markdown("""
    <div class="epistemx-card">
        <h3>🔑 Service Account Authentication</h3>
        <p>This method uses a service account JSON key file for authentication.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Prerequisites")
    st.markdown("""
    1. You must have a Google Cloud Project with Earth Engine API enabled
    2. You must have created a service account with Earth Engine permissions
    3. You must have downloaded the service account JSON key file
    """)
    
    st.markdown("### Authentication Options")
    
    # Tab selection for different input methods
    tab1, tab2, tab3 = st.tabs(["📁 Upload JSON File", "📋 Paste JSON Content", "🔗 Use Existing File"])
    
    # Tab 1: Upload JSON file
    with tab1:
        st.markdown("#### Upload Service Account JSON Key")
        
        uploaded_file = st.file_uploader(
            "Choose your service account JSON file",
            type=['json'],
            help="Upload the JSON key file you downloaded from Google Cloud Console"
        )
        
        project_id_upload = st.text_input(
            "Google Cloud Project ID (optional):",
            key="project_upload",
            help="If not specified, the project ID from the JSON file will be used"
        )
        
        if st.button("🚀 Authenticate with Uploaded File", type="primary", key="auth_upload"):
            if uploaded_file is not None:
                with st.spinner("Authenticating with service account..."):
                    try:
                        # Save uploaded file to temporary location
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                            # Read and validate JSON
                            json_content = json.load(uploaded_file)
                            json.dump(json_content, tmp_file)
                            tmp_file_path = tmp_file.name
                        
                        # Attempt authentication
                        success = initialize_with_service_account(
                            tmp_file_path,
                            project=project_id_upload if project_id_upload else None
                        )
                        
                        # Clean up temp file
                        os.unlink(tmp_file_path)
                        
                        if success:
                            st.session_state.ee_initialized = True
                            st.success("✅ Service account authentication successful!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Authentication failed. Please check your service account file and permissions.")
                            
                    except json.JSONDecodeError:
                        st.error("❌ Invalid JSON file. Please upload a valid service account key file.")
                    except Exception as e:
                        st.error(f"Authentication error: {str(e)}")
            else:
                st.warning("⚠️ Please upload a JSON file first.")
    
    # Tab 2: Paste JSON content
    with tab2:
        st.markdown("#### Paste Service Account JSON Content")
        st.info("Copy and paste the entire content of your service account JSON file below.")
        
        json_content = st.text_area(
            "Service Account JSON:",
            height=300,
            placeholder='{\n  "type": "service_account",\n  "project_id": "your-project",\n  ...\n}',
            help="Paste the complete JSON content from your service account key file"
        )
        
        project_id_paste = st.text_input(
            "Google Cloud Project ID (optional):",
            key="project_paste",
            help="If not specified, the project ID from the JSON content will be used"
        )
        
        if st.button("🚀 Authenticate with JSON Content", type="primary", key="auth_paste"):
            if json_content.strip():
                with st.spinner("Authenticating with service account..."):
                    try:
                        # Validate and parse JSON
                        json_data = json.loads(json_content)
                        
                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                            json.dump(json_data, tmp_file)
                            tmp_file_path = tmp_file.name
                        
                        # Attempt authentication
                        success = initialize_with_service_account(
                            tmp_file_path,
                            project=project_id_paste if project_id_paste else None
                        )
                        
                        # Clean up temp file
                        os.unlink(tmp_file_path)
                        
                        if success:
                            st.session_state.ee_initialized = True
                            st.success("✅ Service account authentication successful!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Authentication failed. Please check your service account credentials and permissions.")
                            
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Invalid JSON format: {str(e)}")
                    except Exception as e:
                        st.error(f"Authentication error: {str(e)}")
            else:
                st.warning("⚠️ Please paste JSON content first.")
    
    # Tab 3: Use existing file
    with tab3:
        st.markdown("#### Use Existing Service Account File")
        st.info("If you have already placed a service account JSON file in the `auth/` directory, you can use it here.")
        
        # Check for existing files in auth directory
        auth_dir = 'auth'
        existing_files = []
        
        if os.path.exists(auth_dir):
            existing_files = [f for f in os.listdir(auth_dir) if f.endswith('.json')]
        
        if existing_files:
            selected_file = st.selectbox(
                "Select service account file:",
                existing_files,
                help="Choose from available JSON files in the auth/ directory"
            )
            
            file_path = os.path.join(auth_dir, selected_file)
            st.code(f"File path: {file_path}", language="text")
            
            project_id_existing = st.text_input(
                "Google Cloud Project ID (optional):",
                key="project_existing",
                help="If not specified, the project ID from the JSON file will be used"
            )
            
            if st.button("🚀 Authenticate with Existing File", type="primary", key="auth_existing"):
                with st.spinner("Authenticating with service account..."):
                    try:
                        success = initialize_with_service_account(
                            file_path,
                            project=project_id_existing if project_id_existing else None
                        )
                        
                        if success:
                            st.session_state.ee_initialized = True
                            st.success("✅ Service account authentication successful!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Authentication failed. Please check your service account file and permissions.")
                            
                    except Exception as e:
                        st.error(f"Authentication error: {str(e)}")
        else:
            st.warning(f"⚠️ No JSON files found in the `{auth_dir}/` directory.")
            st.markdown("""
            **To use this option:**
            1. Create an `auth/` directory in your project root (if it doesn't exist)
            2. Place your service account JSON file in the `auth/` directory
            3. Refresh this page
            """)
    
    # Service account setup instructions
    with st.expander("📖 How to Create a Service Account", expanded=False):
        st.markdown("""
        ### Creating a Service Account for Earth Engine
        
        #### Step 1: Create a Google Cloud Project
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project or select an existing one
        3. Note your Project ID
        
        #### Step 2: Enable Earth Engine API
        1. In the Cloud Console, go to "APIs & Services" > "Library"
        2. Search for "Earth Engine API"
        3. Click "Enable"
        
        #### Step 3: Create Service Account
        1. Go to "IAM & Admin" > "Service Accounts"
        2. Click "Create Service Account"
        3. Enter a name and description
        4. Click "Create and Continue"
        
        #### Step 4: Grant Permissions
        1. Add the role "Earth Engine Resource Admin" or appropriate Earth Engine role
        2. Click "Continue" and then "Done"
        
        #### Step 5: Create and Download Key
        1. Click on the created service account
        2. Go to the "Keys" tab
        3. Click "Add Key" > "Create new key"
        4. Choose "JSON" format
        5. Click "Create" - the JSON file will download automatically
        
        #### Step 6: Register Service Account with Earth Engine
        1. Go to [Earth Engine Code Editor](https://code.earthengine.google.com/)
        2. Register your service account email (found in the JSON file)
        
        For detailed instructions, visit: [Earth Engine Service Account Documentation](https://developers.google.com/earth-engine/guides/service_account)
        """)

# Footer with helpful links
st.divider()
st.markdown("""
<div class="epistemx-card">
    <h4>📚 Helpful Resources</h4>
    <ul>
        <li><a href="https://earthengine.google.com/" target="_blank">Google Earth Engine Homepage</a></li>
        <li><a href="https://developers.google.com/earth-engine/guides/python_install" target="_blank">Earth Engine Python Installation Guide</a></li>
        <li><a href="https://developers.google.com/earth-engine/guides/service_account" target="_blank">Service Account Documentation</a></li>
        <li><a href="https://console.cloud.google.com/" target="_blank">Google Cloud Console</a></li>
    </ul>
</div>
""", unsafe_allow_html=True)
