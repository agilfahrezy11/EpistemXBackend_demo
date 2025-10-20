import streamlit as st
import pandas as pd
from src.src_modul_2 import LULCSchemeClass

# Page configuration
st.set_page_config(
    page_title="Land Cover Classification Scheme",
    page_icon="logos\logo_epistem_crop.png",
    layout="wide"
)

# Page title
st.title("Classification Scheme Definition")
st.divider()
st.markdown("In this module, you need to define the classification scheme that you will be using to generate the land cover map." \
" You have three options to define the classification scheme: Manual input, upload a CSV file, or using a default classfication scheme. The default classification scheme used in this platform are adapted from the RESTORE + project")
st.markdown("---")
#Initialize the manager
#cache resource is used so tha+t for each input of class, the user interface did not run entirely from the top 

@st.cache_resource
def get_lulc_manager():
    return LULCSchemeClass()

manager = get_lulc_manager()
if 'lulc_classes' not in st.session_state:
    st.session_state.lulc_classes = []
if 'lulc_next_id' not in st.session_state:
    st.session_state.lulc_next_id = 1
if 'lulc_edit_mode' not in st.session_state:
    st.session_state.lulc_edit_mode = False
if 'lulc_edit_idx' not in st.session_state:
    st.session_state.lulc_edit_idx = None
#Render the UI
#for this page, a lot of core function is located in the source code, especially for the user inteface to manually input the class
# ===== UI RENDERING SECTION =====
# Tab layout for different input methods
tab1, tab2, tab3 = st.tabs(["➕ Manual Input", "📤 Upload CSV", "📋 Default Scheme"])

# Tab 1: Manual Input
with tab1:
    st.markdown("#### Add a New Class")
    
    col1, col2, col3 = st.columns([1, 3, 2])
    
    with col1:
        if st.session_state.lulc_edit_mode:
            class_id = st.number_input(
                "Class ID",
                value=manager.classes[st.session_state.lulc_edit_idx]['ID'],
                min_value=1,
                step=1,
                key="edit_class_id"
            )
        else:
            class_id = st.number_input(
                "Class ID",
                value=manager.next_id,
                min_value=1,
                step=1,
                key="new_class_id"
            )
    
    with col2:
        if st.session_state.lulc_edit_mode:
            class_name = st.text_input(
                "Class Name",
                value=manager.classes[st.session_state.lulc_edit_idx]['Class Name'],
                placeholder="e.g., Hutan, Permukiman",
                key="edit_class_name"
            )
        else:
            class_name = st.text_input(
                "Class Name",
                value="",
                placeholder="e.g., Hutan, Permukiman",
                key="new_class_name"
            )
    
    with col3:
        if st.session_state.lulc_edit_mode:
            color_code = st.color_picker(
                "Color Code",
                value=manager.classes[st.session_state.lulc_edit_idx]['Color Code'],
                key="edit_color_code"
            )
        else:
            color_code = st.color_picker(
                "Color Code",
                value="#2e8540",
                key="new_color_code"
            )
    
    # Add/Update and Cancel buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    with col_btn1:
        if st.session_state.lulc_edit_mode:
            if st.button("💾 Update Class", type="primary", use_container_width=True):
                if manager.add_class(class_id, class_name, color_code):
                    st.rerun()
        else:
            if st.button("➕ Add Class", type="primary", use_container_width=True):
                if manager.add_class(class_id, class_name, color_code):
                    st.rerun()
    
    with col_btn2:
        if st.session_state.lulc_edit_mode:
            if st.button("❌ Cancel", use_container_width=True):
                manager.cancel_edit()
                st.rerun()

# Tab 2: Upload CSV
with tab2:
    st.markdown("#### Upload Classification Scheme")
    st.info("CSV should contain columns: `ID`, `Class Name`, `Color Code` (or `Color Palette`)")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        key="csv_uploader"
    )
    
    if uploaded_file is not None:
        if st.button("📥 Load CSV", type="primary"):
            if manager.process_csv_upload(uploaded_file):
                st.rerun()

# Tab 3: Default Scheme
with tab3:
    st.markdown("#### Load Default Classification Scheme")
    st.info("Quick start with RESTORE+ project land cover classes")
    
    # Get default schemes (static method, doesn't need manager instance)
    default_schemes = LULCSchemeClass.get_default_schemes()
    
    selected_scheme = st.selectbox(
        "Select a default scheme:",
        options=list(default_schemes.keys())
    )
    
    if st.button("📋 Load Default Scheme", type="primary"):
        manager.load_default_scheme(default_schemes[selected_scheme])
        st.rerun()

# Display current classification scheme
st.markdown("---")
st.markdown("#### Defined Classes")

if not manager.classes:
    st.warning("No classes defined yet. Add your first class above!")
else:
    # Display as DataFrame with color preview
    df_display = pd.DataFrame(manager.classes)
    
    # Create columns for the table
    for idx, row in df_display.iterrows():
        col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 1, 1])
        
        with col1:
            st.write(f"**{row['ID']}**")
        
        with col2:
            st.write(row['Class Name'])
        
        with col3:
            # Color preview
            st.markdown(
                f"<div style='background-color: {row['Color Code']}; "
                f"width: 50px; height: 30px; border: 1px solid #ccc; "
                f"display: inline-block; margin-right: 10px;'></div>"
                f"<code>{row['Color Code']}</code>",
                unsafe_allow_html=True
            )
        
        with col4:
            if st.button("✏️", key=f"edit_{idx}", help="Edit"):
                manager.edit_class(idx)
                st.rerun()
        
        with col5:
            if st.button("🗑️", key=f"delete_{idx}", help="Delete"):
                manager.delete_class(idx)
                st.rerun()
    
    # Download button
    st.markdown("---")
    csv_data = manager.download_csv()
    if csv_data:
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name="classification_scheme.csv",
            mime="text/csv",
            type="primary"
        )
        
        # Show preview
        with st.expander("📋 Preview Classification Scheme"):
            st.dataframe(manager.get_dataframe(), use_container_width=True)


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