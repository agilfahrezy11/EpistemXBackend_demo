import streamlit as st
import pandas as pd
from src.src_modul_2 import LULC_Scheme_Manager

#Page configuration
st.set_page_config(
    page_title="Land Cover Classification Scheme",
    page_icon="logos/logo_epistem_crop.png",
    layout="wide"
)

#Initialize the manager from the source code
@st.cache_resource
def get_lulc_manager():
    return LULC_Scheme_Manager()

manager = get_lulc_manager()

# Page header
st.title("Determining LULC Classification Schema and Classes")
st.divider()

st.markdown("""
In this module, you need to define the classification scheme that you will be using to generate the land cover map.
Three methods are supported in this platform:
- **Manual Input**: Add classes one by one
- **CSV Upload**: Import from existing classification file  
- **Default Scheme**: Use predefined RESTORE+ project classes
""")

st.markdown("---")

#Tab layout for different classification definition
tab1, tab2, tab3 = st.tabs(["➕ Manual Input", "📤 Upload CSV", "📋 Default Scheme"])

#Createa function for manual input the class
def render_manual_input_form():
    """
    Render the manual class input form.
    
    Creates input fields for class ID, name, and color, with support for
    both adding new classes and editing existing ones.
    """
    st.markdown("#### Add a New Class")
    #3 columns
    col1, col2, col3 = st.columns([1, 3, 2])
    
    # Get current values for edit mode
    edit_mode = st.session_state.get('lulc_edit_mode', False)
    edit_idx = st.session_state.get('lulc_edit_idx', None)
    
    if edit_mode and edit_idx is not None:
        current_class = manager.classes[edit_idx]
        default_id = current_class['ID']
        default_name = current_class['Class Name']
        default_color = current_class['Color Code']
        key_suffix = "edit"
    else:
        default_id = manager.next_id
        default_name = ""
        default_color = "#2e8540"
        key_suffix = "new"
    
    with col1:
        class_id = st.number_input(
            "Class ID", 
            value=default_id, 
            min_value=1, 
            step=1,
            key=f"{key_suffix}_class_id"
        )
    
    with col2:
        class_name = st.text_input(
            "Class Name", 
            value=default_name,
            placeholder="e.g., Forest, Settlement",
            key=f"{key_suffix}_class_name"
        )
    
    with col3:
        color_code = st.color_picker(
            "Color Code", 
            value=default_color,
            key=f"{key_suffix}_color_code"
        )
    
    # Action buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    
    with col_btn1:
        button_text = "💾 Update Class" if edit_mode else "➕ Add Class"
        if st.button(button_text, type="primary", width = 'stretch'):
            success, message = manager.add_class(class_id, class_name, color_code)
            if success:
                st.success(f"✅ {message}")
                st.rerun()
            else:
                st.error(f"❌ {message}")
    
    with col_btn2:
        if edit_mode and st.button("❌ Cancel", width = 'stretch'):
            manager.cancel_edit()
            st.rerun()

# Tab 1: Manual Input
with tab1:
    render_manual_input_form()

# Tab 2: Upload CSV
with tab2:
    st.markdown("#### Upload Classification Scheme")
    st.info("""
    **CSV Requirements:**
    - **ID column**: Numeric identifiers (e.g., 'ID', 'Class ID', 'Kode')
    - **Name column**: Class names (e.g., 'Class Name', 'Kelas', 'Name')
    - **Color column** (Optional): Hex color codes (e.g., 'Color', 'Color Code', 'Hex')
    
    If no color column is detected, distinct colors will be automatically assigned.
    """)
    #Code to upload csv
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            #make sure that python can read any form of delimiter
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
            
            #Auto-detect columns
            auto_id, auto_name, auto_color = manager.auto_detect_csv_columns(df)
            
            st.markdown("### Select Columns")
            col1, col2, col3 = st.columns(3)
            #first column to select class ID's column
            with col1:
                id_col = st.selectbox(
                    "Select ID Column *", 
                    df.columns, 
                    index=df.columns.get_loc(auto_id) if auto_id in df.columns else 0
                )
            #second column to select class name's column
            with col2:
                name_col = st.selectbox(
                    "Select Class Name Column *", 
                    df.columns,
                    index=df.columns.get_loc(auto_name) if auto_name in df.columns else 0
                )
            #new column for detecting color palette column
            with col3:
                color_options = ["< No Color Column >"] + list(df.columns)
                default_color_idx = 0
                if auto_color and auto_color in df.columns:
                    default_color_idx = color_options.index(auto_color)
                
                color_col_selection = st.selectbox(
                    "Select Color Column (Optional)",
                    color_options,
                    index=default_color_idx
                )
            #After selection, load the CSV
            if st.button("📤 Load CSV Data", type="primary"):
                color_col = None if color_col_selection == "< No Color Column >" else color_col_selection
                success, message = manager.process_csv_upload(df, id_col, name_col, color_col)
                if success:
                    st.success(f"✅ {message}")
                    # If colors were detected, finalize immediately
                    if color_col:
                        success_final, message_final = manager.finalize_csv_upload()
                        if success_final:
                            st.success(f"✅ {message_final}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
                    
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

    #Color assignment section (only show if no colors were auto-detected)
    if st.session_state.get('csv_temp_classes'):
        st.markdown("---")
        st.markdown("### Adjust Color for each class")
        st.info("Color have been randomly generated. You can adjust them if needed.")
        
        color_assignments = []
        temp_classes = st.session_state.csv_temp_classes
        
        for i, class_data in enumerate(temp_classes):
            col1, col2, col3 = st.columns([1, 3, 2])
            
            with col1:
                st.write(f"**ID: {class_data['ID']}**")
            with col2:
                st.write(f"**{class_data['Class Name']}**")
            with col3:
                color = st.color_picker(
                    f"Color", 
                    value=class_data.get('Color Code', '#2e8540'),
                    key=f"csv_color_{i}",
                    label_visibility="collapsed"
                )
                color_assignments.append(color)
        
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("✅ Finalize Scheme", type="primary", width = 'stretch'):
                success, message = manager.finalize_csv_upload(color_assignments)
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
        
        with col2:
            if st.button("❌ Cancel Upload", width = 'stretch'):
                st.session_state.csv_temp_classes = []
                st.rerun()

# Tab 3: Default Scheme
with tab3:
    st.markdown("#### Load Default Classification Scheme")
    st.info("Quick start with predefined RESTORE+ project land cover classes")
    
    default_schemes = manager.get_default_schemes()
    
    selected_scheme = st.selectbox(
        "Select a default scheme:",
        options=list(default_schemes.keys())
    )
    
    # Preview the selected scheme
    if selected_scheme:
        with st.expander("📋 Preview Classes"):
            preview_df = pd.DataFrame(default_schemes[selected_scheme])
            st.dataframe(preview_df, width = 'stretch')
    
    if st.button("📋 Load Default Scheme", type="primary", width = 'stretch'):
        success, message = manager.load_default_scheme(selected_scheme)
        if success:
            st.success(f"✅ {message}")
            st.rerun()
        else:
            st.error(f"❌ {message}")

#new function to render selected classification scheme from one of the three methods
def render_class_display():
    """
    Render the current classification scheme display.
    
    Shows all defined classes in a table format with color previews,
    edit/delete buttons, and download functionality.
    """
    st.markdown("---")
    st.markdown("#### Current Classification Scheme")

    if not manager.classes:
        st.warning("⚠️ No classes defined yet. Add your first class above!")
        return

    # Display classes in a clean table format
    for idx, class_data in enumerate(manager.classes):
        col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 1, 1])
        
        with col1:
            st.write(f"**{class_data['ID']}**")
        
        with col2:
            st.write(class_data['Class Name'])
        
        with col3:
            # Color preview with code
            st.markdown(
                f"""<div style='display: flex; align-items: center;'>
                    <div style='background-color: {class_data['Color Code']}; 
                                width: 40px; height: 25px; border: 1px solid #ccc; 
                                margin-right: 8px; border-radius: 3px;'></div>
                    <code>{class_data['Color Code']}</code>
                </div>""",
                unsafe_allow_html=True
            )
        
        with col4:
            if st.button("✏️", key=f"edit_{idx}", help="Edit class"):
                manager.edit_class(idx)
                st.rerun()
        
        with col5:
            if st.button("🗑️", key=f"delete_{idx}", help="Delete class"):
                success, message = manager.delete_class(idx)
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()

    # Download and preview section
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        csv_data = manager.get_csv_data()
        if csv_data:
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name="classification_scheme.csv",
                mime="text/csv",
                type="primary",
                width = 'stretch'
            )
    
    with col2:
        with st.expander("📋 Preview Data"):
            st.dataframe(manager.get_dataframe(), width = 'stretch')

# Render the class display
render_class_display()

def render_navigation():
    """
    Render module navigation and completion status.
    
    Provides navigation buttons to previous/next modules and displays
    completion status based on whether classes have been defined.
    """
    st.divider()
    
    # Store classification data for other modules
    if manager.classes:
        st.session_state['classification_df'] = manager.get_dataframe()
    
    # Module completion check
    module_completed = len(manager.classes) > 0
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ Back to Module 1", width = 'stretch'):
            st.switch_page("pages/1_Module_1_Generate_Image_Mosaic.py")
    
    with col2:
        if module_completed:
            if st.button("➡️ Go to Module 3: Training data generation", 
                        type="primary", width = 'stretch'):
                st.switch_page("pages/3_Module_3_Generate_ROI.py")
        else:
            st.button("🔒 Complete Module 2 First", 
                     disabled=True, width = 'stretch',
                     help="Add at least one class to proceed")
    
    # Status indicator
    if module_completed:
        st.success(f"✅ Module completed with {len(manager.classes)} classes")
    else:
        st.info("💡 Add at least one class to complete this module")

# Render navigation
render_navigation()