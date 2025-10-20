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
if "lulc_manager" not in st.session_state:
    st.session_state.lulc_manager = LULCSchemeClass()

manager = st.session_state.lulc_manager
# Session state is handled by the manager initialization
# No need for duplicate initialization here

#User interface for manual input, upload CSV, or using a default classification scheme
#Tab layout for different options
tab1, tab2, tab3 = st.tabs(["➕ Manual Input", "📤 Upload CSV", "📋 Default Scheme"])

#Tab 1: Manual Input the class
with tab1:
    st.markdown("#### Add a New Class")
    #define the column
    col1, col2, col3 = st.columns([1, 3, 2])
    #first column
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
    #Second column
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
    #Third column
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
    
    #Add/Update and cancel buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    with col_btn1:
        if st.session_state.lulc_edit_mode:
            if st.button("💾 Update Class", type="primary",  width='stretch'):
                if manager.add_class(class_id, class_name, color_code):
                    st.rerun()
        else:
            if st.button("➕ Add Class", type="primary",  width='stretch'):
                if manager.add_class(class_id, class_name, color_code):
                    st.rerun()
    
    with col_btn2:
        if st.session_state.lulc_edit_mode:
            if st.button("❌ Cancel",  width='stretch'):
                manager.cancel_edit()
                st.rerun()

# Tab 2: Upload CSV
with tab2:
    st.markdown("#### Upload Classification Scheme")
    st.info("📝 CSV Requirements:\n"
            "- **Required**: ID column (e.g., 'ID', 'Class ID', 'Kode')\n"
            "- **Required**: Class Name column (e.g., 'Class Name', 'Kelas', 'Name')\n\n"
            "After uploading, you'll be able to assign colors to each class individually.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="csv_uploader")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
        
        # --- AUTO-DETECT COLUMN NAMES ---
        columns_lower = [c.lower().replace(" ", "").replace("_", "") for c in df.columns]

        def auto_find(keywords):
            for kw in keywords:
                for i, col in enumerate(columns_lower):
                    if kw in col:
                        return df.columns[i]
            return None

        auto_id = auto_find(["id", "classid", "kode"])
        auto_name = auto_find(["classname", "class", "kelas", "name"])

        # --- USER DROPDOWNS (MANUAL OVERRIDE) ---
        st.write("### Step 2: Map Columns")
        id_col = st.selectbox("Select ID Column *", df.columns, index=df.columns.get_loc(auto_id) if auto_id in df.columns else 0)
        name_col = st.selectbox("Select Class Name Column *", df.columns, index=df.columns.get_loc(auto_name) if auto_name in df.columns else 0)

        # --- LOAD CSV BUTTON ---
        if st.button("📤 Load CSV Data", type="primary"):
            if manager.process_csv_upload(df, id_col, name_col):
                st.success("CSV data loaded successfully! Now assign colors to each class below.")
                st.rerun()
            else:
                st.error("Failed to load CSV. Check the column mapping and try again.")

    # --- COLOR ASSIGNMENT SECTION ---
    if 'csv_temp_classes' in st.session_state and st.session_state.csv_temp_classes:
        st.markdown("---")
        st.markdown("### Step 3: Assign Colors to Classes")
        st.info("Choose a color for each class from your CSV file:")
        
        color_assignments = []
        
        # Create color pickers for each class
        for i, class_data in enumerate(st.session_state.csv_temp_classes):
            col1, col2, col3 = st.columns([1, 3, 2])
            
            with col1:
                st.write(f"**ID: {class_data['ID']}**")
            
            with col2:
                st.write(f"**{class_data['Class Name']}**")
            
            with col3:
                color = st.color_picker(
                    f"Color for {class_data['Class Name']}", 
                    value="#2e8540",
                    key=f"csv_color_{i}"
                )
                color_assignments.append(color)
        
        # Finalize button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("✅ Finalize Classification Scheme", type="primary", width='stretch'):
                if manager.finalize_csv_upload(color_assignments):
                    st.success("Classification scheme created successfully!")
                    # Clear the temporary data
                    if 'csv_temp_classes' in st.session_state:
                        del st.session_state.csv_temp_classes
                    st.rerun()
        
        with col2:
            if st.button("❌ Cancel Upload",  width='stretch'):
                if 'csv_temp_classes' in st.session_state:
                    del st.session_state.csv_temp_classes
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
                manager.EditClass(idx)
                st.rerun()
        
        with col5:
            if st.button("🗑️", key=f"delete_{idx}", help="Delete"):
                manager.DeleteClass(idx)
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
            st.dataframe(manager.get_dataframe(),  width='stretch')


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
module_2_completed = len(manager.classes) > 0
# Create two columns for navigation buttons
col1, col2 = st.columns(2)

with col1:
    # Back to Module 1 button (always available)
    if st.button("⬅️ Back to Module 1: Data Upload",  width='stretch'):
        st.switch_page("pages/1_Module_1_Data_Upload.py")

with col2:
    # Forward to Module 3 button (conditional)
    if module_2_completed:
        if st.button("➡️ Go to Module 4: Analyze ROI", type="primary",  width='stretch'):
            st.switch_page("pages/3_Module_4_Analyze_ROI.py")
    else:
        st.button("🔒 Complete Module 2 First", disabled=True,  width='stretch', 
                 help="Please add at least one class to the classification scheme")

# Optional: Show completion status
if module_2_completed:
    st.success(f"✅ Classification scheme completed with {len(manager.classes)} classes")
else:
    st.info("Add classification scheme to complete this module")