import streamlit as st
from epistemx.shapefile_utils import shapefile_validator, EE_converter
from epistemx.module_7 import Thematic_Accuracy_Assessment
import pandas as pd
import numpy as np
import geemap.foliumap as geemap
import tempfile
import zipfile
import os
import geopandas as gpd
import plotly.express as px
from epistemx.ee_config import initialize_earth_engine
initialize_earth_engine()

#Page configuration
st.set_page_config(
    page_title="Thematic Accuracy Assessment",
    page_icon="logos/logo_epistem_crop.png",
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

# Apply custom theme
load_css()


#Initialize accuracy assessment manager
@st.cache_resource
def get_accuracy_manager():
    return Thematic_Accuracy_Assessment()

manager = get_accuracy_manager()

# Page header
st.title("Thematic Accuracy Assessment")
st.divider()

st.markdown("""
Evaluate the thematic accuracy of your land cover classification from Module 6 using independent validation data. 
In order to run this module you need an ground reference data containing class ID and class name similar to a ROI
The accuracy of land cover map is evaluated using a confusion matrix, with the following key metrics

- **Overall Accuracy** with confidence intervals
- **Kappa Coefficient** for agreement assessment  
- **F1-Score** for class-level performance
""")

st.markdown("---")

#This module wont run if classification result from module 6 is not avaliable
def check_prerequisites():
    """Check if required data from previous modules is available"""
    if 'classification_result' not in st.session_state or st.session_state.classification_result is None:
        st.error("❌ No classification result found from Module 6.")
        st.warning("Please complete Module 6 first to generate a land cover classification map.")
        st.stop()
    else:
        st.success("✅ Classification map loaded from Module 6")
        return st.session_state.classification_result

#Initialize the functions
lcmap = check_prerequisites()
#function to upload the ground reference data (similar to module 3, but in a function)
def process_shapefile_upload(uploaded_file):
    """Process uploaded shapefile and convert to Earth Engine format"""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploaded file
            zip_path = os.path.join(tmpdir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract zip file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            # Find shapefile
            shp_files = []
            for root, _, files in os.walk(tmpdir):
                for fname in files:
                    if fname.lower().endswith(".shp"):
                        shp_files.append(os.path.join(root, fname))

            if not shp_files:
                return False, "No .shp file found in the uploaded zip.", None, None

            # Read and process shapefile
            gdf = gpd.read_file(shp_files[0])
            
            #Validate and clean geometry using the helper module 
            validator = shapefile_validator(verbose=False)
            converter = EE_converter(verbose=False)
            
            gdf_cleaned = validator.validate_and_fix_geometry(gdf, geometry="mixed")
            
            if gdf_cleaned is None:
                return False, "Geometry validation failed", None, None
            
            # Convert to Earth Engine format using helper module
            ee_data = converter.convert_roi_gdf(gdf_cleaned)
            
            if ee_data is None:
                return False, "Failed to convert to Google Earth Engine format", None, None
            
            return True, "Validation data processed successfully", ee_data, gdf_cleaned
            
    except Exception as e:
        return False, f"Error processing shapefile: {str(e)}", None, None

#similar to module 1 but wrap in function
def render_validation_upload():
    """Render validation data upload section"""
    st.subheader("Step 1: Upload Ground Reference Data")
    st.info("📁 Upload a **.zip shapefile** containing your independent validation samples with class IDs.")

    uploaded_file = st.file_uploader("Choose a zipped shapefile (.zip)", type=["zip"])

    if uploaded_file:
        with st.spinner("Processing validation data..."):
            success, message, ee_data, gdf_cleaned = process_shapefile_upload(uploaded_file)
            
            if success:
                st.success(f"✅ {message}")
                
                # Store in session state
                st.session_state['validation_data'] = ee_data
                st.session_state['validation_gdf'] = gdf_cleaned
                
                # Show data preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(gdf_cleaned.head(), use_container_width=True)
                
                # Show map preview
                st.markdown("**📍 Validation Points Distribution:**")
                centroid = gdf_cleaned.geometry.centroid.iloc[0]
                preview_map = geemap.Map(center=[centroid.y, centroid.x], zoom=8)
                preview_map.add_geojson(gdf_cleaned.__geo_interface__, layer_name="Validation Points")
                preview_map.to_streamlit(height=500)
                
            else:
                st.error(f"❌ {message}")
                if "Make sure your shapefile includes" not in message:
                    st.info("💡 Ensure your shapefile includes all necessary files (.shp, .shx, .dbf, .prj)")

#run validation data upload
render_validation_upload()
#Function for definining user input for accuracy assessment
def user_input_for_accuracy_assessment():
    """Render accuracy assessment configuration and execution"""
    st.divider()
    st.subheader("Step 2: Configure and Run Assessment")

    if "validation_data" not in st.session_state or st.session_state.validation_data is None:
        st.warning("⚠️ Please upload your validation data first.")
        return

    gdf_cleaned = st.session_state.get('validation_gdf')
    if gdf_cleaned is None:
        st.error("Validation data not properly loaded.")
        return

    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        class_prop = st.selectbox(
            "Class ID Field:",
            options=gdf_cleaned.columns.tolist(),
            index=gdf_cleaned.columns.get_loc("CLASS_ID") if "CLASS_ID" in gdf_cleaned.columns else 0,
            help="Field containing numeric class identifiers (e.g., 1, 2, 3, 4)"
        )
    
    with col2:
        scale = st.number_input(
            "Pixel Size (meters):",
            min_value=10,
            max_value=1000,
            value=30,
            help="Spatial resolution for sampling the classified map"
        )

    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        confidence = st.slider(
            "Confidence Level for Accuracy Intervals:",
            min_value=0.90,
            max_value=0.99,
            value=0.95,
            step=0.01,
            format="%.2f"
        )

    # Run assessment
    if st.button("🎯 Evaluate Map Accuracy", type="primary", use_container_width=True):
        with st.spinner("Running thematic accuracy assessment..."):
            success, results = manager.run_accuracy_assessment(
                lcmap=lcmap,
                validation_data=st.session_state.validation_data,
                class_property=class_prop,
                scale=scale,
                confidence=confidence
            )

            if success:
                st.session_state["accuracy_results"] = results
                st.success("✅ Thematic accuracy assessment completed!")
                st.rerun()
            else:
                st.error(f"❌ Assessment failed: {results.get('error', 'Unknown error')}")

#run user inpur function
user_input_for_accuracy_assessment()

#Function to display accuracy assessment
def render_accuracy_results():
    """Render accuracy assessment results"""
    #if not initialize yet
    if "accuracy_results" not in st.session_state:
        return
    #Then initialze the session state to store the accuracy
    results = st.session_state["accuracy_results"]
    
    if 'error' in results:
        st.error(f"❌ {results['error']}")
        return

    st.divider()
    st.subheader("Accuracy Assessment Results")

    #Prepared to display the key result
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Accuracy", 
            f"{results['overall_accuracy']*100:.2f}%"
        )
    
    with col2:
        st.metric(
            "Kappa Coefficient", 
            f"{results['kappa']:.3f}"
        )
    
    with col3:
        ci = results['overall_accuracy_ci']
        confidence_pct = int(results['confidence_level'] * 100)
        st.metric(
            f"{confidence_pct}% Confidence Interval", 
            f"{ci[0]*100:.1f}% - {ci[1]*100:.1f}%"
        )
    
    with col4:
        st.metric(
            "Sample Size", 
            f"{results['n_total']} points"
        )

    # Interpretation guidelines
    with st.expander("📖 Panduan Interpretasi Hasil", expanded=False):
        st.markdown("""
        **Overall Accuracy (Akurasi Keseluruhan):**
        - **≥ 85%**: Akurasi yang baik untuk sebagian besar kajian penutup/penggunaan lahan
        - **70-84%**: Akurasi sedang, dapat digunakan untuk kajian tertentu 
        - **< 70%**: Akurasi rendah, disarankan untuk memperbaiki klasifikasi
        
        **Kappa Coefficient:**
        - **≥ 0.8**: Kesepakatan yang kuat antara peta dan data referensi
        - **0.6-0.79**: Kesepakatan sedang antara peta dan data referensi
        - **< 0.6**: Kesepakatan lemah antara peta dan data referensi
        
        **F1-Score:**
        - **Nilai mendekati 1.0**: Performa yang baik untuk kelas tersebut
        - **Nilai mendekati 0.5**: Performa sedang untuk kelas tersebut
        - **Nilai mendekati 0.0**: Performa rendah untuk kelas tersebut
        
        **Confidence Interval:**
        - Menunjukkan rentang kepercayaan untuk akurasi keseluruhan
        - Interval yang sempit menunjukkan estimasi yang lebih presisi
        
        💡 **Catatan:** Interpretasi ini bersifat umum. Standar akurasi dapat bervariasi tergantung studi yang dilakukan dan kompleksitas skema klasifikasi.
        """)
    
    st.markdown("---")

    # Class-level metrics table
    st.markdown("---")
    st.subheader("Class-Level Performance")
    st.markdown("""
    Akurasi pada tingkat kelas dapat digunakan untuk menilai kualitas peta pada kelas tertentu. 
    Metrik akurasi yang digunakan untuk menilai akurasi pada tingkat kelas:
    - **Producer's Accuracy**:  Akurasi ini menjawab pertanyaan 'Seberapa baik peta memetakan kelas yang ada di lapangan?'.
    Metrik ini memberikan informasi mengenai kesalahan omisi, yaitu ketika data dari kelas yang benar tidak terdeteksi atau terlewat oleh peta.
    - **User's Accuracy**: Akurasi ini menjawab pertanyaan 'Seberapa dipercayanya hasil klasifikasi kelas tertentu?'
    Metrik ini memberikan informasi mengenai kesalahan komisi, yaitu ketika peta melakukan kesalahan klasifikasi dengan memasukkan data dari kelas lain ke dalam kelas tersebut.
    """)

    # Get class names from Module 2 if available
    class_names = []
    if 'lulc_classes_final' in st.session_state:
        # Create a mapping from class ID to class name
        class_id_to_name = {}
        for cls in st.session_state['lulc_classes_final']:
            class_id = cls.get('ID', cls.get('Class ID'))
            class_name = cls.get('Class Name', cls.get('Land Cover Class', f'Class {class_id}'))
            class_id_to_name[class_id] = class_name
        
        # Create class names list
        for i in range(len(results["producer_accuracy"])):
            if i in class_id_to_name:
                class_names.append(class_id_to_name[i])
            else:
                class_names.append(f"Class {i}")
    else:
        # Fallback to generic class names if Module 2 data not available
        class_names = [f"Class {i}" for i in range(len(results["producer_accuracy"]))]

    df_metrics = pd.DataFrame({
        "Class ID": range(len(results['producer_accuracy'])),
        "Class Name": class_names,
        "Producer's Accuracy (%)": [round(v * 100, 2) for v in results['producer_accuracy']],
        "User's Accuracy (%)": [round(v * 100, 2) for v in results['user_accuracy']],
        "F1-Score (%)": [round(v * 100, 2) for v in results["f1_scores"]],
    })
    
    st.dataframe(df_metrics, use_container_width=True)

    # Confusion matrix visualization
    st.markdown("---")
    st.subheader("🔄 Confusion Matrix")
    st.markdown("Shows how often each class was correctly identified vs confused with other classes")

    cm_array = results["confusion_matrix"]
    n_classes = len(cm_array)
    
    # Get class names from Module 2 if available
    class_labels = []
    if 'lulc_classes_final' in st.session_state:
        # Create a mapping from class ID to class name
        class_id_to_name = {}
        for cls in st.session_state['lulc_classes_final']:
            class_id = cls.get('ID', cls.get('Class ID'))
            class_name = cls.get('Class Name', cls.get('Land Cover Class', f'Class {class_id}'))
            class_id_to_name[class_id] = class_name
        
        # Create labels for confusion matrix (ID: Name format)
        for i in range(n_classes):
            if i in class_id_to_name:
                class_labels.append(f"{i}: {class_id_to_name[i]}")
            else:
                class_labels.append(f"Class {i}")
    else:
        # Fallback to generic class labels if Module 2 data not available
        class_labels = [f"Class {i}" for i in range(n_classes)]
    
    cm_df = pd.DataFrame(
        cm_array,
        columns=[f"Predicted {label}" for label in class_labels],
        index=[f"Actual {label}" for label in class_labels]
    )
    
    # Calculate dynamic height based on number of classes
    base_height = max(500, n_classes * 60)  # Minimum 500px, 60px per class
    
    # Create heatmap
    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Confusion Matrix: Actual vs Predicted Classes"
    )
    
    # Improve layout for better readability
    fig.update_layout(
        height=base_height,
        width=None,  # Let it use container width
        title={
            'text': "Confusion Matrix: Actual vs Predicted Classes",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis={
            'tickangle': 45,
            'side': 'bottom'
        },
        yaxis={
            'tickangle': 0
        },
        font=dict(size=10),
        margin=dict(l=150, r=50, t=80, b=150)  # Add margins for labels
    )
    
    # Update text annotations for better visibility
    fig.update_traces(
        texttemplate="%{z}",
        textfont={"size": max(8, 14 - n_classes)},  # Smaller text for more classes
        hovertemplate="<b>Actual:</b> %{y}<br><b>Predicted:</b> %{x}<br><b>Count:</b> %{z}<extra></extra>"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add interpretation help
    with st.expander("📖 How to Read the Confusion Matrix"):
        st.markdown("""
        **Understanding the Confusion Matrix:**
        - **Rows (Actual)**: True class labels from your validation data
        - **Columns (Predicted)**: Classes predicted by your classification map
        - **Diagonal values**: Correct predictions (higher is better)
        - **Off-diagonal values**: Misclassifications (lower is better)
        
        **Perfect Classification**: All values would be on the diagonal with zeros elsewhere.
        
        **Common Issues to Look For:**
        - High off-diagonal values indicate confusion between specific classes
        - Consistently low values in a row suggest the map struggles to detect that class
        - Consistently high values in a column suggest the map over-predicts that class
        """)
    
    # Add summary statistics
    st.markdown("#### Confusion Matrix Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate per-class accuracy (diagonal / row sum)
        cm_array = np.array(results["confusion_matrix"])
        row_sums = cm_array.sum(axis=1)
        diagonal = np.diag(cm_array)
        per_class_accuracy = np.divide(diagonal, row_sums, out=np.zeros_like(diagonal, dtype=float), where=row_sums!=0) * 100
        
        accuracy_df = pd.DataFrame({
            "Class": class_labels,
            "Correct Predictions": diagonal,
            "Total Samples": row_sums,
            "Class Accuracy (%)": np.round(per_class_accuracy, 1)
        })
        
        st.markdown("**Per-Class Performance:**")
        st.dataframe(accuracy_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Show most confused classes
        st.markdown("**Most Common Misclassifications:**")
        misclass_data = []
        
        for i in range(len(cm_array)):
            for j in range(len(cm_array)):
                if i != j and cm_array[i][j] > 0:  # Off-diagonal elements
                    misclass_data.append({
                        'Actual': class_labels[i],
                        'Predicted': class_labels[j], 
                        'Count': cm_array[i][j]
                    })
        
        if misclass_data:
            misclass_df = pd.DataFrame(misclass_data)
            misclass_df = misclass_df.sort_values('Count', ascending=False).head(5)
            st.dataframe(misclass_df, use_container_width=True, hide_index=True)
        else:
            st.success("🎉 Perfect classification! No misclassifications found.")

    # Export results option
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Create downloadable results
        results_summary = {
            'Overall_Accuracy_Percent': results['overall_accuracy'] * 100,
            'Kappa_Coefficient': results['kappa'],
            'Confidence_Interval_Lower': results['overall_accuracy_ci'][0] * 100,
            'Confidence_Interval_Upper': results['overall_accuracy_ci'][1] * 100,
            'Sample_Size': results['n_total'],
            'Scale_Meters': results['scale']
        }
        
        results_df = pd.DataFrame([results_summary])
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Results Summary",
            data=csv_data,
            file_name="accuracy_assessment_results.csv",
            mime="text/csv",
           use_container_width=True
        )
    
    with col2:
        # Download detailed metrics
        detailed_csv = df_metrics.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Class Metrics",
            data=detailed_csv,
            file_name="class_level_accuracy.csv",
            mime="text/csv",
           use_container_width=True
        )

def render_navigation():
    """Render navigation options"""
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⬅️ Back to Module 6", use_container_width=True):
            st.switch_page("pages/5_Module_6_Classification_and_LULC_Creation.py")
    
    with col2:
        st.info("💡 Return to Module 6 to improve your classification model if needed")

# Render results and navigation
render_accuracy_results()
render_navigation()