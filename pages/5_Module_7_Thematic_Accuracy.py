import streamlit as st
from src.module_helpers import init_gee, shapefile_validator, EE_converter
init_gee()
from src.src_modul_7 import Accuracy_Assessment
import pandas as pd
import geemap.foliumap as geemap
import tempfile
import zipfile
import traceback
import os
import geopandas as gpd

#Page configuration
st.set_page_config(
    page_title="Accuracy Assessment",
    page_icon="logos\logo_epistem_crop.png",
    layout="wide"
)
#Set the page title (for the canvas)
st.title("Thematic Accuracy Assessment")
st.divider()
st.markdown("""
This module evaluates the accuracy of the **categorical land cover data** generated in Module 6. The accuracy assessment procudure follows the same workflow as module 6, with the main difference is
            being the data being tested. The Assessment is calculated using independent **ground reference (validation) data, which you need to prepare before hand**.
            For assessing the quality of the land cover data, three main metrics is use""")
st.markdown("1. Overall Accuracy, with confidence interval")
st.markdown("2. Kappa Coefficient")
st.markdown("3. F1-score")

#User input, Ground Reference Data
st.subheader("Upload Ground Reference Data (shapefile)")
st.markdown("currently the platform only support shapefile in .zip format")
# ==================== CHECK PREREQUISITES ====================
if 'classification_result' not in st.session_state or st.session_state.classification_result is None:
    st.error("❌ No classification result found from Module 6.")
    st.warning("Please complete Module 6 first to generate a land cover classification map.")
    st.stop()
else:
    lcmap = st.session_state.classification_result
    st.success("✅ Classification map loaded from Module 6")

# ==================== UPLOAD VALIDATION DATA ====================
st.subheader("Step 1: Upload Ground Reference Data")
st.markdown("Upload a **.zip shapefile** containing your independent validation samples.")

uploaded_file = st.file_uploader("Upload a zipped shapefile (.zip)", type=["zip"])
ref_data = None

#define AOI upload function
if uploaded_file:
    #Extract the uploaded zip file to a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # write uploaded bytes to disk (required before reading zip)
        zip_path = os.path.join(tmpdir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        # Find the .shp file in the extracted files (walk subfolders)
        shp_files = []
        for root, _, files in os.walk(tmpdir):
            for fname in files:
                if fname.lower().endswith(".shp"):
                    shp_files.append(os.path.join(root, fname))

        if len(shp_files) == 0:
            st.error("No .shp file found in the uploaded zip.")
        else:
            try:
                # Read the shapefile
                gdf = gpd.read_file(shp_files[0])
                st.success("ROI loaded successfully!")
                validate = shapefile_validator(verbose=False)
                converter = EE_converter(verbose=False)
                st.markdown("ROI table preview:")
                st.write(gdf)
                # Validate and fix geometry
                gdf_cleaned = validate.validate_and_fix_geometry(gdf, geometry="mixed")
                
                if gdf_cleaned is not None:
                    # Convert to EE geometry safely
                    ref_data = converter.convert_roi_gdf(gdf_cleaned)
                    
                    if ref_data is not None:
                        st.success("ROI conversion completed!")
                        
                        # Show a small preview map centered on AOI
                        # Store in session state
                        st.session_state['validation_data'] = ref_data
                        st.session_state['validation_gdf'] = gdf_cleaned
                        st.text("Region of Interest distribution:")
                        centroid = gdf_cleaned.geometry.centroid.iloc[0]
                        preview_map = geemap.Map(center=[centroid.y, centroid.x], zoom=6)
                        preview_map.add_geojson(gdf_cleaned.__geo_interface__, layer_name="AOI")
                        preview_map.to_streamlit(height=600)
                    else:
                        st.error("Failed to convert ROI to Google Earth Engine format")
                else:
                    st.error("Geometry validation failed")
                    
            except Exception as e:
                st.error(f"Error reading shapefile: {e}")
                st.info("Make sure your shapefile includes all necessary files (.shp, .shx, .dbf, .prj)")
                st.code(traceback.format_exc())
# ==================== ACCURACY ASSESSMENT ====================
st.divider()
st.subheader("Step 2: Run Accuracy Assessment")

if "validation_data" not in st.session_state or st.session_state.validation_data is None:
    st.warning("⚠️ Please upload your validation data first.")
else:
    class_prop = st.selectbox(
        "Select the field containing numeric class IDs, (example: 1, 2, 3, 4, etc):",
        options=gdf_cleaned.columns.tolist(),
        index=gdf_cleaned.columns.get_loc("CLASS_ID") if "CLASS_ID" in gdf_cleaned.columns else 0,
        key="class_property"
    )
    scale = st.number_input(
        "Pixel Size (m)",
        min_value=10,
        max_value=1000,
        value=30,
        help="Spatial resolution for sampling classified map"
    )

    if st.button("Evaluate Map Accuracy", type="primary"):
        with st.spinner("Running thematic accuracy assessment..."):
            try:
                assess = Accuracy_Assessment()
                results = assess.thematic_assessment(
                    lcmap=lcmap,
                    validation_data=st.session_state.validation_data,
                    class_property=class_prop,
                    scale=scale
                )

                st.session_state["accuracy_results"] = results
                st.success("✅ Thematic accuracy assessment completed!")

            except Exception as e:
                st.error(f"Error running accuracy assessment: {e}")
                st.code(traceback.format_exc())

# ==================== DISPLAY RESULTS ====================
if "accuracy_results" in st.session_state:
    acc = st.session_state["accuracy_results"]
    st.subheader("Accuracy Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy", f"{acc['overall_accuracy']*100:.2f}%")
    col2.metric("Kappa Coefficient", f"{acc['kappa']:.3f}")
    if "overall_accuracy_ci" in acc:
        ci = acc["overall_accuracy_ci"]
        col3.metric("95% CI (Overall Accuracy)", f"{ci[0]*100:.2f}% - {ci[1]*100:.2f}%")

    st.markdown("---")
    st.subheader("Class-Level Metrics")

    df_metrics = pd.DataFrame({
        "Class ID": range(len(acc['producer_accuracy'])),
        "Producer's Accuracy (Recall) (%)": [round(v * 100, 2) for v in acc['producer_accuracy']],
        "User's Accuracy (Precision) (%)": [round(v * 100, 2) for v in acc['user_accuracy']],
        "F1-score (%)": [round(v * 100, 2) for v in acc["f1_scores"]],
    })
    st.dataframe(df_metrics, use_container_width=True)

    st.markdown("---")
    st.subheader("Confusion Matrix")

    cm = pd.DataFrame(
        acc["confusion_matrix"],
        columns=[f"Pred_{i}" for i in range(len(acc["confusion_matrix"]))],
        index=[f"Actual_{i}" for i in range(len(acc["confusion_matrix"]))]
    )
    import plotly.express as px
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, width='stretch')

st.divider()
st.markdown("Return to **Module 6** if you want to re-run or improve your classification model.")