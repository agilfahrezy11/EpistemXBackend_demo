import streamlit as st
from src.src_modul_2 import LULCSchemeClass

# Page configuration
st.set_page_config(
    page_title="Land Cover Classification Scheme",
    page_icon="logos\logo_epistem_crop.png",
    layout="wide"
)

#Initialize the manager
@st.cache_resource
def get_lulc_manager():
    return LULCSchemeClass()

manager = get_lulc_manager()

# Page title
st.title("Classification Scheme Definition")
st.divider()
st.markdown("In this module, you need to define the classification scheme that you will be using to generate the land cover map." \
" You have three options to define the classification scheme: Manual input, upload a CSV file, or using a default classfication scheme. The default classification scheme used in this platform are adapted from the RESTORE + project")
st.markdown("---")
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
st.divider()
st.subheader("Module Navigation")
# Check if Module 2 is completed (has at least one class)
module_2_completed = len(st.session_state.get("classes", [])) > 0
# Create two columns for navigation buttons
col1, col2 = st.columns(2)

with col1:
    # Back to Module 1 button (always available)
    if st.button("⬅️ Back to Module 2: Classification Scheme", use_container_width=True):
        st.switch_page("pages/2_Module_2_Classification_scheme.py")

with col2:
    # Forward to Module 3 button (conditional)
    if module_2_completed:
        if st.button("➡️ Go to Module 4: Analyze ROI", type="primary", use_container_width=True):
            st.switch_page("pages/3_Module_4_Analyze_ROI.py")
    else:
        st.button("🔒 Complete Module 2 First", disabled=True, use_container_width=True, 
                 help="Please add at least one class to the classification scheme")

# Optional: Show completion status
if module_2_completed:
    st.success(f"✅ Classification scheme completed with {len(st.session_state['classes'])} classes")
else:
    st.info("Add classification scheme to complete this module")