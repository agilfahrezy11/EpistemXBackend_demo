import streamlit as st
import geemap.foliumap as geemap
import geopandas as gpd
from src.src_modul_1 import Reflectance_Data, Reflectance_Stats
from src.utils_shapefile_validation_conversion import shapefile_validator, EE_converter
import tempfile
import zipfile
import os
import ee
import datetime
ee.Authenticate()
ee.Initialize()

#title of the module
st.title("Search and Generate Landsat Image Mosaic")
st.divider()
#module name
markdown = """
This module allows users to search and generate a Landsat image mosaic for a specified area of interest (AOI) and time range using Google Earth Engine (GEE) data.
"""
#set page layout and side info
st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "logos\logo_epistem.png"
st.sidebar.image(logo)

#Based on early experiments, shapefile with complex geometry often cause issues in GEE
#The following functions are used to handle the common geometry issues

#User input, AOI upload
st.subheader("Upload Area of Interest (Shapefile)")
st.markdown("currently the platform only support shapefile in .zip format")
uploaded_file = st.file_uploader("Upload a zipped shapefile (.zip)", type=["zip"])
aoi = None

#define AOI upload function
if uploaded_file:
    # Extract the uploaded zip file to a temporary directory
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
                st.success("Shapefile loaded successfully!")
                validate = shapefile_validator(verbose=False)
                converter = EE_converter(verbose=False)
                # Validate and fix geometry
                gdf_cleaned = validate.validate_and_fix_geometry(gdf)
                
                if gdf_cleaned is not None:
                    # Convert to EE geometry safely
                    aoi = converter.convert_aoi_gdf(gdf_cleaned)
                    
                    if aoi is not None:
                        st.success("AOI conversion completed!")
                        
                        # Show a small preview map centered on AOI
                        st.text("Area of interest preview:")
                        centroid = gdf_cleaned.geometry.centroid.iloc[0]
                        preview_map = geemap.Map(center=[centroid.y, centroid.x], zoom=7)
                        preview_map.add_geojson(gdf_cleaned.__geo_interface__, layer_name="AOI")
                        preview_map.to_streamlit(height=500)
                    else:
                        st.error("Failed to convert AOI to Google Earth Engine format")
                else:
                    st.error("Geometry validation failed")
                    
            except Exception as e:
                st.error(f"Error reading shapefile: {e}")
                st.info("Make sure your shapefile includes all necessary files (.shp, .shx, .dbf, .prj)")

st.divider()
#User input, search criteria
st.subheader("Specify Imagery Search Criteria")
st.text("Enter the acquisition date range, cloud cover percentage, and Landsat mission type. " \
"Current platform support Landsat 1,2 and 4 at sensor radiance and Landsat 5-9 Collection 2 Surface Reflectance Analysis Ready Data (ARD), excluding the thermal bands. Landsat mission avaliability is as follows:")
st.markdown("1. Landsat 1 Multispectral Scanner/MSS (1972 - 1978)")
st.markdown("2. Landsat 2 Multispectral Scanner/MSS (1978 - 1982)")
st.markdown("3. Landsat 4 Thematic Mapper/TM (1982 - 1993)")
st.markdown("4. Landsat 5 Thematic Mapper/TM (1984 - 2012)")
st.markdown("5. Landsat 7 Enhanced Thematic Mapper Plus/ETM+ (1999 - 2021)")
st.markdown("6. Landsat 8 Operational Land Imager/OLI (2013 - present)")
st.markdown("7. Landsat 9 Operational Land Imager-2/OLI-2 (2021 - present)")
#specified the avaliable sensor type
sensor_type = ['L5_SR', 'L7_SR', 'L8_SR', 'L9_SR']
#create a selection box for sensor type
sensor_dict = {
    "Landsat 1 MSS": "L1_RAW",
    "Landsat 2 MSS": "L2_RAW",
    "Landsat 4 TM": "L4_SR",
    "Landsat 5 TM": "L5_SR",
    "Landsat 7 ETM+": "L7_SR",
    "Landsat 8 OLI": "L8_SR",
    "Landsat 9 OLI-2": "L9_SR"
}
sensor_names = list(sensor_dict.keys())
#user define parameters for the search
#select sensor type
selected_sensor_name = st.selectbox("Select Landsat Sensor:", sensor_names, index=2)
optical_data = sensor_dict[selected_sensor_name]  # This is what you pass to your backend
#Date selection
#Year only
st.subheader("Select Time Period")
date_mode = st.radio(
    "Choose date selection mode:",
    ["Select by year", "Custom date range"],
    index=0
)

if date_mode == "Select by year":
    # Just year input
    years = list(range(1972, datetime.datetime.now().year + 1))
    years.reverse()  #Newest First

    year = st.selectbox("Select Year", years, index=years.index(2020))
    start_date = str(year)
    end_date = str(year)
#Full date
else:
    # Full date inputs
    default_start = datetime.date(2020, 1, 1)
    default_end = datetime.date(2020, 12, 31)
    start_date_dt = st.date_input("Start Date:", default_start)
    end_date_dt = st.date_input("End Date:", default_end)
    start_date = start_date_dt.strftime("%Y-%m-%d")
    end_date = end_date_dt.strftime("%Y-%m-%d")

#cloud cover slider
cloud_cover = st.slider("Maximum Cloud Cover (%):", 0, 100, 30)

#Search the landsat imagery
if st.button("Search Landsat Imagery") and aoi:
    reflectance = Reflectance_Data()
    collection, meta = reflectance.get_optical_data(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        optical_data=optical_data,
        cloud_cover=cloud_cover,
        verbose=False,
        compute_detailed_stats=False
    )
    stats = Reflectance_Stats()
    detailed_stats = stats.get_collection_statistics(collection, compute_stats=True, print_report=True)
    st.success(f"Found {detailed_stats['total_images']} images.")
        # Debug: check collection size (safe server-side call)
    try:
        coll_size = int(collection.size().getInfo())
    except Exception as e:
        st.error(f"Failed to query collection size: {e}")
        coll_size = 0
    st.write(f"Collection size: {coll_size}")

    if coll_size == 0:
        st.warning("No images found for the selected criteria, increase cloud cover threshold or change the date range.")

    #get valid pixels (number of cloudless pixel in date range)
    valid_px = collection.reduce(ee.Reducer.count()).clip(aoi)
    stats = valid_px.reduceRegion(
    reducer=ee.Reducer.minMax().combine(
        reducer2=ee.Reducer.mean(), sharedInputs=True
    ),
    geometry=aoi,
    scale=30,
    maxPixels=1e13
    ).getInfo()
    #Display the search information as report
    summary_md = f"""
    ### Landsat Imagery Search Summary

    - **Total Images Found:** {detailed_stats.get('total_images', 'N/A')}
    - **Date Range of Images:** {detailed_stats.get('date_range', 'N/A')}
    - **Unique WRS Tiles:** {detailed_stats.get('unique_tiles', 'N/A')}
    - **Path Row Tiles:** {detailed_stats.get('path_row_tiles', 'N/A')}
    - **Scene IDs:** {', '.join(detailed_stats.get('Scene_ids', [])) if detailed_stats.get('Scene_ids') else 'N/A'} 
    - **Image acquisition dates:** {', '.join(detailed_stats.get('individual_dates', [])) if detailed_stats.get('individual_dates') else 'N/A'}
    - **Average Scene Cloud Cover:** {detailed_stats.get('cloud_cover', {}).get('mean', 'N/A')}%
    - **Date Range:** {detailed_stats.get('date_range', 'N/A')}
    - **Cloud Cover Range:** {detailed_stats.get('cloud_cover', {}).get('min', 'N/A')}% - {detailed_stats.get('cloud_cover', {}).get('max', 'N/A')}%
    """
    st.markdown(summary_md)
    # Optionally, display the full stats as a table
    #st.subheader("Detailed Statistics") {'bands': ['RED', 'GREEN', 'BLUE'], 'min': 0, 'max': 0.3}
    #st.write(detailed_stats)
    if detailed_stats['total_images'] > 0:
        #visualization parameters
        vis_params = {
        'min': 0,
        'max': 0.4,
        'gamma': [0.95, 1.1, 1],
        'bands':['NIR', 'RED', 'GREEN']
        }

        #Create and image composite/mosaic
        composite = collection.median().clip(aoi)
        # Store in session state for use in other modules
        st.session_state['composite'] = composite
        st.session_state['Image_metadata'] = detailed_stats
        st.session_state['AOI'] = aoi
        st.session_state['visualization'] = vis_params
        # Display the image using geemap
        centroid = gdf.geometry.centroid.iloc[0]
        m = geemap.Map(center=[centroid.y, centroid.x], zoom=6)
        m.addLayer(composite, vis_params, 'Landsat Mosaic', shown= True)
        m.addLayer(collection, vis_params, 'Landsat Collection', shown=True)
        m.add_geojson(gdf.__geo_interface__, layer_name="AOI", shown = False)
        m.to_streamlit(height=600)
else:
    st.info("Upload an AOI and specify search criteria to begin.")
#Link the next page
st.divider()
st.subheader("Module Navigation")
if 'composite' in st.session_state:
    if st.button("Go to Module 2: Classification Scheme"):
        st.switch_page("pages/2_Module_2_Classification_scheme.py")
else:
    st.button("🔒 Complete Module 1 First", disabled=True)
    st.info("Please Generate an imagery mosaic before proceeding")