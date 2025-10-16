import streamlit as st
from src.src_modul_2 import LULCSchemeClass

# Page configuration
st.set_page_config(
    page_title="Land Cover Classification Scheme",
    page_icon="🗺️",
    layout="wide"
)

# Initialize the manager
@st.cache_resource
def get_lulc_manager():
    return LULCSchemeClass()

manager = get_lulc_manager()

# Page title
st.title("🌍 Land Cover Mapping Application")
st.subheader("Module 2: Classification Scheme Definition")

# Render the UI
manager.render_ui()

# Optional: Access the data in other parts of your app
if st.checkbox("Show classification data for Module 3"):
    st.markdown("### Data for Next Module")
    df = manager.get_dataframe()
    st.dataframe(df)
    
    # You can pass this DataFrame to Module 3
    if len(df) > 0:
        st.success(f"✅ Ready to use {len(df)} classes in Module 3!")
        
        # Example: Store in session state for use in other pages
        st.session_state['classification_df'] = df