# Duplicating REMAP Authentication in Streamlit

## Overview

This guide shows how to replicate REMAP's dual authentication system in a Streamlit application:

1. **Service Account** - For Earth Engine API access (backend)
2. **OAuth2** - For Google Drive exports (user-specific)

Streamlit presents unique challenges compared to the Vue.js + Flask architecture, but the core authentication patterns remain the same.

---

## Key Differences: REMAP vs Streamlit

| Aspect | REMAP (Vue + Flask) | Streamlit |
|--------|---------------------|-----------|
| **Architecture** | Separate frontend/backend | Single Python app |
| **Session Management** | GAE Datastore | Streamlit session_state |
| **OAuth Flow** | JavaScript gapi + backend | Python only (no JS) |
| **Background Tasks** | GAE Task Queue | Threading or async |
| **State Persistence** | Server-side sessions | In-memory session_state |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ STREAMLIT APP STRUCTURE                                          │
└─────────────────────────────────────────────────────────────────┘

streamlit_app.py
    │
    ├─ Service Account Auth (on startup)
    │   └─ ee.Initialize(service_account_credentials)
    │
    ├─ OAuth2 User Auth (on demand)
    │   ├─ Generate authorization URL
    │   ├─ User clicks link → Google consent
    │   ├─ Callback receives code
    │   └─ Exchange code for tokens
    │
    ├─ Session State Management
    │   ├─ st.session_state['credentials']
    │   ├─ st.session_state['user_email']
    │   └─ st.session_state['export_status']
    │
    └─ UI Components
        ├─ Map display
        ├─ Training point collection
        ├─ Classification
        └─ Drive export
```

---

## Part 1: Service Account Setup (Earth Engine)

### Step 1: Install Dependencies

```bash
pip install streamlit earthengine-api google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

### Step 2: Create Service Account Files

**File: `secrets/ee_service_account.json`**

```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "your-service-account@your-project.iam.gserviceaccount.com",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token"
}
```

### Step 3: Initialize Earth Engine

**File: `streamlit_app.py`**

```python
import streamlit as st
import ee
import json

# Initialize Earth Engine with service account (runs once)
@st.cache_resource
def initialize_earth_engine():
    """Initialize Earth Engine with service account credentials."""
    try:
        # Load service account credentials
        with open('secrets/ee_service_account.json') as f:
            service_account_info = json.load(f)
        
        credentials = ee.ServiceAccountCredentials(
            service_account_info['client_email'],
            'secrets/ee_service_account.json'
        )
        
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"Failed to initialize Earth Engine: {e}")
        return False

# Initialize on app startup
ee_initialized = initialize_earth_engine()
```

**Key Points:**
- `@st.cache_resource` ensures initialization happens only once
- Service account credentials are loaded from JSON file
- No user interaction required
- All Earth Engine operations use these credentials automatically

---

## Part 2: OAuth2 Setup (Google Drive)

### Step 1: Create OAuth2 Credentials

**File: `secrets/oauth_client_secret.json`**

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

**Important:** Add `http://localhost:8501` to authorized redirect URIs in Google Cloud Console.

### Step 2: OAuth2 Flow Implementation

**File: `auth_utils.py`**

```python
import streamlit as st
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import json
import os

# OAuth2 scopes
SCOPES = [
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/userinfo.email'
]

def get_oauth_flow():
    """Create OAuth2 flow object."""
    flow = Flow.from_client_secrets_file(
        'secrets/oauth_client_secret.json',
        scopes=SCOPES,
        redirect_uri='http://localhost:8501'
    )
    return flow

def get_authorization_url():
    """Generate authorization URL for user to visit."""
    flow = get_oauth_flow()
    auth_url, state = flow.authorization_url(
        access_type='offline',  # Get refresh token
        include_granted_scopes='true',
        prompt='consent'  # Force consent screen to get refresh token
    )
    return auth_url, state

def exchange_code_for_credentials(auth_code):
    """Exchange authorization code for credentials."""
    try:
        flow = get_oauth_flow()
        flow.fetch_token(code=auth_code)
        credentials = flow.credentials
        
        # Get user info
        service = build('oauth2', 'v2', credentials=credentials)
        user_info = service.userinfo().get().execute()
        
        return credentials, user_info
    except Exception as e:
        st.error(f"Failed to exchange code: {e}")
        return None, None

def credentials_to_dict(credentials):
    """Convert credentials to dictionary for storage."""
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

def dict_to_credentials(creds_dict):
    """Convert dictionary back to credentials object."""
    return Credentials(
        token=creds_dict['token'],
        refresh_token=creds_dict['refresh_token'],
        token_uri=creds_dict['token_uri'],
        client_id=creds_dict['client_id'],
        client_secret=creds_dict['client_secret'],
        scopes=creds_dict['scopes']
    )

def refresh_credentials(credentials):
    """Refresh expired credentials."""
    from google.auth.transport.requests import Request
    
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    return credentials
```

### Step 3: Streamlit OAuth2 UI

**File: `streamlit_app.py` (continued)**

```python
import streamlit as st
from auth_utils import *

def render_auth_section():
    """Render authentication UI."""
    st.sidebar.title("🔐 Authentication")
    
    # Check if user is authenticated
    if 'credentials' not in st.session_state:
        st.sidebar.warning("Not signed in to Google Drive")
        
        # Generate authorization URL
        if st.sidebar.button("Sign in with Google"):
            auth_url, state = get_authorization_url()
            st.session_state['oauth_state'] = state
            st.sidebar.markdown(f"[Click here to authorize]({auth_url})")
            st.sidebar.info("After authorizing, paste the code below:")
        
        # Input for authorization code
        auth_code = st.sidebar.text_input("Authorization Code", type="password")
        
        if auth_code:
            credentials, user_info = exchange_code_for_credentials(auth_code)
            if credentials:
                st.session_state['credentials'] = credentials_to_dict(credentials)
                st.session_state['user_email'] = user_info['email']
                st.sidebar.success(f"Signed in as {user_info['email']}")
                st.rerun()
    else:
        # User is authenticated
        st.sidebar.success(f"✅ Signed in as {st.session_state['user_email']}")
        
        if st.sidebar.button("Sign Out"):
            del st.session_state['credentials']
            del st.session_state['user_email']
            st.rerun()

# Call in main app
render_auth_section()
```

---

## Part 3: Complete Streamlit Application

**File: `streamlit_app.py` (full example)**

```python
import streamlit as st
import ee
import json
from auth_utils import *
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from io import BytesIO
import time

# Page config
st.set_page_config(
    page_title="Earth Engine Classifier",
    page_icon="🌍",
    layout="wide"
)

# ============================================================================
# PART 1: SERVICE ACCOUNT - EARTH ENGINE INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_earth_engine():
    """Initialize Earth Engine with service account credentials."""
    try:
        with open('secrets/ee_service_account.json') as f:
            service_account_info = json.load(f)
        
        credentials = ee.ServiceAccountCredentials(
            service_account_info['client_email'],
            'secrets/ee_service_account.json'
        )
        
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"Failed to initialize Earth Engine: {e}")
        return False

# Initialize Earth Engine
ee_initialized = initialize_earth_engine()

if not ee_initialized:
    st.stop()

# ============================================================================
# PART 2: OAUTH2 - GOOGLE DRIVE AUTHENTICATION
# ============================================================================

def render_auth_section():
    """Render authentication UI in sidebar."""
    st.sidebar.title("🔐 Google Drive Auth")
    
    if 'credentials' not in st.session_state:
        st.sidebar.warning("Not signed in")
        
        if st.sidebar.button("Sign in with Google"):
            auth_url, state = get_authorization_url()
            st.session_state['oauth_state'] = state
            
            # Display clickable link
            st.sidebar.markdown(f"### [🔗 Click to Authorize]({auth_url})")
            st.sidebar.info("After authorizing, paste the code from the URL below:")
            st.sidebar.code("http://localhost:8501/?code=YOUR_CODE&scope=...")
        
        # Input for authorization code
        auth_code = st.sidebar.text_input("Paste authorization code", type="password")
        
        if auth_code:
            with st.spinner("Authenticating..."):
                credentials, user_info = exchange_code_for_credentials(auth_code)
                if credentials:
                    st.session_state['credentials'] = credentials_to_dict(credentials)
                    st.session_state['user_email'] = user_info['email']
                    st.sidebar.success(f"Signed in as {user_info['email']}")
                    st.rerun()
    else:
        st.sidebar.success(f"✅ {st.session_state['user_email']}")
        
        if st.sidebar.button("Sign Out"):
            del st.session_state['credentials']
            del st.session_state['user_email']
            st.rerun()

# ============================================================================
# PART 3: EARTH ENGINE OPERATIONS (USES SERVICE ACCOUNT)
# ============================================================================

@st.cache_data
def load_predictor_image():
    """Load predictor image with all bands."""
    # Climate data
    bioclim = ee.Image('WORLDCLIM/V1/BIO').select(
        ['bio01', 'bio12'],
        ['Mean_Annual_Temperature', 'Annual_Precipitation']
    )
    
    # Elevation
    elevation = ee.Image('USGS/SRTMGL1_003').rename(['Elevation'])
    slope = ee.Terrain.slope(elevation).rename(['Slope'])
    
    # Landsat (you'll need to create your own composite or use public data)
    # For demo, using a simple Landsat 8 composite
    landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterDate('2020-01-01', '2020-12-31') \
        .median() \
        .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5'], 
                ['Blue', 'Green', 'Red', 'NIR'])
    
    # Calculate indices
    ndvi = landsat.normalizedDifference(['NIR', 'Red']).rename(['NDVI'])
    ndwi = landsat.normalizedDifference(['Green', 'NIR']).rename(['NDWI'])
    
    # Combine all bands
    return ee.Image([bioclim, elevation, slope, landsat, ndvi, ndwi])

def classify_region(training_points, predictors, region):
    """Classify region using training points."""
    # Load predictor image
    composite = load_predictor_image().select(predictors)
    
    # Create training FeatureCollection
    features = []
    for class_data in training_points:
        label = class_data['label']
        for point in class_data['points']:
            feature = ee.Feature(
                ee.Geometry.Point([point['lng'], point['lat']]),
                {'label': label}
            )
            features.append(feature)
    
    train_fc = ee.FeatureCollection(features)
    
    # Sample predictors at training points
    training = composite.reduceRegions(
        collection=train_fc,
        reducer=ee.Reducer.first(),
        scale=30
    )
    
    # Train classifier
    classifier = ee.Classifier.smileRandomForest(100).train(
        features=training,
        classProperty='label',
        inputProperties=composite.bandNames()
    )
    
    # Classify
    classified = composite.classify(classifier).clip(region)
    
    return classified

def get_map_tiles(image, vis_params):
    """Get map tile URL for visualization."""
    map_id = image.getMapId(vis_params)
    return map_id['tile_fetcher'].url_format

# ============================================================================
# PART 4: GOOGLE DRIVE EXPORT (USES USER OAUTH2)
# ============================================================================

def export_to_drive(classified_image, region, filename):
    """Export classified image to user's Google Drive."""
    
    # Check if user is authenticated
    if 'credentials' not in st.session_state:
        st.error("Please sign in to Google Drive first")
        return False
    
    try:
        # Get user credentials
        creds_dict = st.session_state['credentials']
        credentials = dict_to_credentials(creds_dict)
        
        # Refresh if expired
        credentials = refresh_credentials(credentials)
        
        # Update stored credentials
        st.session_state['credentials'] = credentials_to_dict(credentials)
        
        # Start Earth Engine export task
        task = ee.batch.Export.image.toDrive(
            image=classified_image,
            description=filename,
            folder='REMAP_Exports',
            fileNamePrefix=filename,
            scale=30,
            region=region.getInfo()['coordinates'],
            maxPixels=1e10,
            fileFormat='GeoTIFF'
        )
        
        task.start()
        
        # Store task ID in session
        st.session_state['export_task_id'] = task.id
        st.session_state['export_status'] = 'RUNNING'
        
        return True
        
    except Exception as e:
        st.error(f"Export failed: {e}")
        return False

def check_export_status():
    """Check status of export task."""
    if 'export_task_id' not in st.session_state:
        return None
    
    task_id = st.session_state['export_task_id']
    
    # Get task status
    tasks = ee.batch.Task.list()
    for task in tasks:
        if task.id == task_id:
            status = task.status()['state']
            st.session_state['export_status'] = status
            return status
    
    return None

# ============================================================================
# PART 5: MAIN UI
# ============================================================================

def main():
    st.title("🌍 Earth Engine Land Cover Classifier")
    st.markdown("Uses **Service Account** for Earth Engine + **OAuth2** for Drive exports")
    
    # Render authentication section
    render_auth_section()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📍 Training Data", "🗺️ Classification", "💾 Export"])
    
    with tab1:
        st.header("Collect Training Points")
        
        # Initialize training data in session state
        if 'training_classes' not in st.session_state:
            st.session_state['training_classes'] = []
        
        # Add class
        col1, col2 = st.columns(2)
        with col1:
            class_name = st.text_input("Class Name", "Forest")
        with col2:
            class_label = st.number_input("Class Label", 1, 10, 1)
        
        if st.button("Add Class"):
            st.session_state['training_classes'].append({
                'name': class_name,
                'label': class_label,
                'points': []
            })
        
        # Display classes
        for i, cls in enumerate(st.session_state['training_classes']):
            st.write(f"**{cls['name']}** (Label: {cls['label']}) - {len(cls['points'])} points")
        
        # In a real app, you'd integrate a map widget here
        st.info("💡 In production, integrate folium or pydeck for interactive point collection")
    
    with tab2:
        st.header("Run Classification")
        
        # Select predictors
        available_predictors = [
            'NDVI', 'NDWI', 'Elevation', 'Slope',
            'Mean_Annual_Temperature', 'Annual_Precipitation'
        ]
        
        selected_predictors = st.multiselect(
            "Select Predictors",
            available_predictors,
            default=['NDVI', 'NDWI', 'Elevation']
        )
        
        if st.button("Classify Region"):
            if len(st.session_state.get('training_classes', [])) < 2:
                st.error("Need at least 2 classes")
            else:
                with st.spinner("Classifying..."):
                    # Define region (example)
                    region = ee.Geometry.Rectangle([-122.5, 37.5, -122.0, 38.0])
                    
                    # Classify
                    classified = classify_region(
                        st.session_state['training_classes'],
                        selected_predictors,
                        region
                    )
                    
                    # Store in session
                    st.session_state['classified_image'] = classified
                    st.session_state['region'] = region
                    
                    st.success("Classification complete!")
                    
                    # Get map tiles
                    vis_params = {
                        'min': 1,
                        'max': len(st.session_state['training_classes']),
                        'palette': ['green', 'brown', 'blue']
                    }
                    
                    tile_url = get_map_tiles(classified, vis_params)
                    st.write(f"Map tiles: {tile_url}")
    
    with tab3:
        st.header("Export to Google Drive")
        
        if 'classified_image' not in st.session_state:
            st.warning("Run classification first")
        elif 'credentials' not in st.session_state:
            st.warning("Sign in to Google Drive first")
        else:
            filename = st.text_input("Export Filename", "remap_export")
            
            if st.button("Export to Drive"):
                with st.spinner("Starting export..."):
                    success = export_to_drive(
                        st.session_state['classified_image'],
                        st.session_state['region'],
                        filename
                    )
                    
                    if success:
                        st.success("Export started! Check status below.")
            
            # Check export status
            if 'export_task_id' in st.session_state:
                st.subheader("Export Status")
                
                if st.button("Refresh Status"):
                    status = check_export_status()
                    
                    if status == 'COMPLETED':
                        st.success("✅ Export completed! Check your Google Drive.")
                    elif status == 'RUNNING':
                        st.info("⏳ Export in progress...")
                    elif status == 'FAILED':
                        st.error("❌ Export failed")
                    else:
                        st.warning(f"Status: {status}")

if __name__ == "__main__":
    main()
```

---

## Part 4: Handling OAuth2 Callback

### Challenge: Streamlit Doesn't Have Built-in Callback Handling

**Solution 1: Use Query Parameters (Simplest)**

```python
# In streamlit_app.py

# Check for OAuth callback in URL
query_params = st.query_params

if 'code' in query_params:
    auth_code = query_params['code']
    
    # Exchange code for credentials
    credentials, user_info = exchange_code_for_credentials(auth_code)
    
    if credentials:
        st.session_state['credentials'] = credentials_to_dict(credentials)
        st.session_state['user_email'] = user_info['email']
        
        # Clear query params
        st.query_params.clear()
        st.rerun()
```

**Solution 2: Use Streamlit-OAuth Component (Recommended)**

```bash
pip install streamlit-oauth
```

```python
from streamlit_oauth import OAuth2Component

# Create OAuth2 component
oauth2 = OAuth2Component(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    authorize_endpoint="https://accounts.google.com/o/oauth2/auth",
    token_endpoint="https://oauth2.googleapis.com/token",
    refresh_token_endpoint="https://oauth2.googleapis.com/token",
    revoke_token_endpoint="https://oauth2.googleapis.com/revoke",
)

# Authorize button
result = oauth2.authorize_button(
    name="Sign in with Google",
    icon="https://www.google.com/favicon.ico",
    redirect_uri="http://localhost:8501",
    scope="https://www.googleapis.com/auth/drive.file",
    key="google_oauth",
    extras_params={"access_type": "offline", "prompt": "consent"}
)

if result and 'token' in result:
    st.session_state['credentials'] = result
    st.success("Signed in!")
```

---

## Part 5: Advanced Features

### 1. Persistent Session Storage

Streamlit's session state is in-memory only. For persistence:

```python
import pickle
import os

def save_session():
    """Save session to disk."""
    session_data = {
        'credentials': st.session_state.get('credentials'),
        'user_email': st.session_state.get('user_email')
    }
    
    with open('.streamlit_session.pkl', 'wb') as f:
        pickle.dump(session_data, f)

def load_session():
    """Load session from disk."""
    if os.path.exists('.streamlit_session.pkl'):
        with open('.streamlit_session.pkl', 'rb') as f:
            session_data = pickle.load(f)
            
        for key, value in session_data.items():
            st.session_state[key] = value

# Call on startup
load_session()
```

### 2. Background Export Monitoring

```python
import threading
import time

def monitor_export_task(task_id):
    """Monitor export task in background."""
    while True:
        tasks = ee.batch.Task.list()
        for task in tasks:
            if task.id == task_id:
                status = task.status()['state']
                st.session_state['export_status'] = status
                
                if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
                    return
        
        time.sleep(10)

# Start monitoring thread
if st.button("Export to Drive"):
    # ... start export ...
    
    thread = threading.Thread(
        target=monitor_export_task,
        args=(task.id,),
        daemon=True
    )
    thread.start()
```

### 3. Automatic Credential Refresh

```python
def ensure_valid_credentials():
    """Ensure credentials are valid, refresh if needed."""
    if 'credentials' not in st.session_state:
        return False
    
    creds_dict = st.session_state['credentials']
    credentials = dict_to_credentials(creds_dict)
    
    # Check if expired
    if credentials.expired and credentials.refresh_token:
        from google.auth.transport.requests import Request
        credentials.refresh(Request())
        
        # Update stored credentials
        st.session_state['credentials'] = credentials_to_dict(credentials)
    
    return True

# Call before Drive operations
if ensure_valid_credentials():
    # Proceed with Drive export
    pass
```

---

## Part 6: Deployment Considerations

### Local Development

```bash
streamlit run streamlit_app.py
```

### Streamlit Cloud Deployment

**File: `.streamlit/secrets.toml`**

```toml
[ee_service_account]
type = "service_account"
project_id = "your-project"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"

[oauth_client]
client_id = "YOUR_CLIENT_ID.apps.googleusercontent.com"
client_secret = "YOUR_CLIENT_SECRET"
```

**Access secrets in code:**

```python
import streamlit as st

# Service account
service_account_info = st.secrets["ee_service_account"]

# OAuth client
client_id = st.secrets["oauth_client"]["client_id"]
client_secret = st.secrets["oauth_client"]["client_secret"]
```

### Update Redirect URI

For Streamlit Cloud, update redirect URI to:
```
https://your-app-name.streamlit.app
```

---

## Part 7: Complete File Structure

```
streamlit_earth_engine_app/
│
├── streamlit_app.py          # Main application
├── auth_utils.py              # OAuth2 utilities
├── ee_utils.py                # Earth Engine utilities
├── requirements.txt           # Dependencies
│
├── secrets/                   # Local secrets (gitignored)
│   ├── ee_service_account.json
│   └── oauth_client_secret.json
│
├── .streamlit/
│   ├── config.toml           # Streamlit config
│   └── secrets.toml          # Secrets for cloud deployment
│
└── .gitignore
```

**File: `requirements.txt`**

```
streamlit>=1.28.0
earthengine-api>=0.1.380
google-auth>=2.23.0
google-auth-oauthlib>=1.1.0
google-auth-httplib2>=0.1.1
google-api-python-client>=2.100.0
```

**File: `.gitignore`**

```
secrets/
.streamlit_session.pkl
__pycache__/
*.pyc
.env
```

---

## Part 8: Key Differences from REMAP

### 1. No JavaScript OAuth Flow

**REMAP:** Uses `gapi.auth2` JavaScript library  
**Streamlit:** Pure Python OAuth flow with redirect

### 2. Session Management

**REMAP:** Server-side sessions in GAE Datastore  
**Streamlit:** In-memory `st.session_state` (lost on refresh)

**Solution:** Implement persistent storage or use cookies

### 3. Background Tasks

**REMAP:** GAE Task Queue for long-running exports  
**Streamlit:** Threading or async (limited by single-process nature)

**Solution:** Use external task queue (Celery, Cloud Tasks) for production

### 4. Real-time Updates

**REMAP:** Polling with `setTimeout`  
**Streamlit:** Manual refresh or `st.rerun()` with intervals

**Solution:** Use `st.empty()` with periodic updates

---

## Part 9: Testing the Flow

### Test Service Account (Earth Engine)

```python
import ee

# Initialize
credentials = ee.ServiceAccountCredentials(
    'your-service-account@project.iam.gserviceaccount.com',
    'secrets/ee_service_account.json'
)
ee.Initialize(credentials)

# Test
image = ee.Image('USGS/SRTMGL1_003')
print("Elevation bands:", image.bandNames().getInfo())
```

### Test OAuth2 (Google Drive)

```python
from auth_utils import get_authorization_url, exchange_code_for_credentials

# Get auth URL
auth_url, state = get_authorization_url()
print(f"Visit: {auth_url}")

# After user authorizes, paste code
code = input("Enter code: ")

# Exchange
credentials, user_info = exchange_code_for_credentials(code)
print(f"Signed in as: {user_info['email']}")
```

---

## Summary

### ✅ What Works Well in Streamlit

- Service account authentication (identical to REMAP)
- Earth Engine operations (identical to REMAP)
- Simple OAuth2 flow (with some adjustments)
- Single-file deployment

### ⚠️ Challenges

- No built-in OAuth callback handling (need workarounds)
- Session state not persistent (need external storage)
- Limited background task support (need external queue)
- No real-time updates without manual refresh

### 🎯 Recommended Approach

1. **Development:** Use the simple query parameter approach
2. **Production:** Use `streamlit-oauth` component or external auth service
3. **Persistence:** Store credentials in encrypted database or cloud storage
4. **Background Tasks:** Use Cloud Functions or Cloud Run for exports

### 📚 Next Steps

1. Set up service account and register with Earth Engine
2. Create OAuth2 credentials in Google Cloud Console
3. Implement basic Streamlit app with authentication
4. Add Earth Engine classification logic
5. Implement Drive export with progress tracking
6. Deploy to Streamlit Cloud

The core authentication patterns from REMAP translate well to Streamlit, with the main differences being in the OAuth callback handling and session persistence.
