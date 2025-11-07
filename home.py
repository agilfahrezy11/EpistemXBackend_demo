import streamlit as st
import leafmap.foliumap as leafmap
from epistemx import auto_initialize
import os
import tempfile

st.set_page_config(layout="wide")

# Load custom CSS
def load_css():
    """Load custom CSS for EpistemX theme"""
    try:
        with open('.streamlit/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # CSS file not found, continue without custom styling

# Apply custom theme
load_css()

# Set up temporary directory for leafmap
if not os.path.exists('/tmp'):
    os.makedirs('/tmp', exist_ok=True)
os.environ['TMPDIR'] = '/tmp'

# Initialize Earth Engine once when the app starts
if 'ee_initialized' not in st.session_state:
    try:
        result = auto_initialize()
        st.session_state.ee_initialized = result
        if result:
            st.success("Earth Engine initialized successfully!")
        else:
            st.error("Failed to initialize Earth Engine. Please check your authentication.")
    except Exception as e:
        st.error(f"Earth Engine initialization error: {e}")
        st.session_state.ee_initialized = False

# Customize the sidebar
markdown = """
An working example module 1 and 3 of Epistem land cover mapping platform. Adapted from:
<https://github.com/opengeos/streamlit-map-template>
"""

st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "logos/logo_epistem.png"
st.sidebar.image(logo)

# Customize page title with branded header
st.markdown("""
<div class="main-header">
    <h1>🛰️ EpistemX Land Cover Mapping Platform</h1>
    <p>Advanced Earth Observation Data Processing & Analysis</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="epistemx-card">
    <h3>🌍 Welcome to EpistemX</h3>
    <p>This multipage platform demonstrates EpistemX's powerful land cover mapping capabilities, 
    featuring automated Landsat imagery processing for your area of interest.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="module-header">📋 Instructions</div>', unsafe_allow_html=True)

st.markdown("""
<div class="epistemx-card">
    <h4>🚀 Getting Started</h4>
    <ol>
        <li><strong>Define Area of Interest:</strong> Draw a rectangle on the map or upload a shapefile (zip)</li>
        <li><strong>Set Acquisition Date:</strong> Specify the year - images will be filtered from January 1 to December 31</li>
        <li><strong>Configure Parameters:</strong> Set cloud cover percentage and sensor type (Landsat 5 TM - Landsat 9 OLI2)</li>
        <li><strong>Generate Mosaic:</strong> Click run to create your satellite imagery mosaic</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Status indicator
if st.session_state.get('ee_initialized', False):
    st.markdown('<div class="success-message">✅ Earth Engine Ready - You can proceed with analysis</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="warning-message">⚠️ Earth Engine Not Initialized - Please check authentication</div>', unsafe_allow_html=True)

m = leafmap.Map(center = [-5.003394, 113.598633], zoom = 5)
m.add_basemap("OpenTopoMap")
m.to_streamlit(height=500)
