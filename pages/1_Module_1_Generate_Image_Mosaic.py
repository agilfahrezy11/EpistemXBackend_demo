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
from src.src_utils_initializer import EarthEngineAuth

if not EarthEngineAuth.initialize():
    st.stop()
# Page configuration
st.set_page_config(
    page_title="Module 1",
    page_icon="logos\logo_epistem_crop.png",
    layout="wide"
)
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
# Initialize session state for storing collection and composite
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'composite' not in st.session_state:
    st.session_state.composite = None
if 'aoi' not in st.session_state:
    st.session_state.aoi = None
if 'gdf' not in st.session_state:
    st.session_state.gdf = None
if 'export_tasks' not in st.session_state:
    st.session_state.export_tasks = []
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
                        st.session_state.aoi = aoi
                        st.session_state.gdf = gdf_cleaned
                        
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
#Dump some information for supported imagery in the platform
st.text("Enter the acquisition date range, cloud cover percentage, and Landsat mission type. " \
"Current platform support Landsat 1-3 at sensor radiance and Landsat 4-9 Collection 2 Surface Reflectance Analysis Ready Data (ARD), excluding the thermal bands. Landsat mission avaliability is as follows:")
st.markdown("1. Landsat 1 Multispectral Scanner/MSS (1972 - 1978)")
st.markdown("2. Landsat 2 Multispectral Scanner/MSS (1978 - 1982)")
st.markdown("3. Landsat 3 Multispectral Scanner/MSS (1978 - 1983)")
st.markdown("3. Landsat 4 Thematic Mapper/TM (1982 - 1993)")
st.markdown("4. Landsat 5 Thematic Mapper/TM (1984 - 2012)")
st.markdown("5. Landsat 7 Enhanced Thematic Mapper Plus/ETM+ (1999 - 2021)")
st.markdown("6. Landsat 8 Operational Land Imager/OLI (2013 - present)")
st.markdown("7. Landsat 9 Operational Land Imager-2/OLI-2 (2021 - present)")
st.markdown("Spatial resolution for Landsat 1-3 is 60 m, while the rest of them have the spatial resolution of 30 m")
#specified the avaliable sensor type
#sensor_type = ['L5_SR', 'L7_SR', 'L8_SR', 'L9_SR']
#create a selection box for sensor type
sensor_dict = {
    "Landsat 1 MSS": "L1_RAW",
    "Landsat 2 MSS": "L2_RAW",
    "Landsat 3 MSS": "L3_RAW",
    "Landsat 4 TM": "L4_SR",
    "Landsat 5 TM": "L5_SR",
    "Landsat 7 ETM+": "L7_SR",
    "Landsat 8 OLI": "L8_SR",
    "Landsat 9 OLI-2": "L9_SR"
}
sensor_names = list(sensor_dict.keys())
#user define parameters for the search
#select sensor type
selected_sensor_name = st.selectbox("Select Landsat Sensor:", sensor_names, index=7)
optical_data = sensor_dict[selected_sensor_name]  #passing to backend process
#Date selection
#Year only
st.subheader("Select Time Period")
st.markdown("If 'select by year' is choosen, the system automatically search imagery from January 1 untill December 31 ")
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
if st.button("Search Landsat Imagery", type="primary") and st.session_state.aoi is not None:
    with st.spinner("Searching for Landsat imagery..."):
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
        
        #Store the metadata for export
        st.session_state.search_metadata = {
            'sensor': optical_data,
            'start_date': start_date,
            'end_date': end_date,
            'num_images': detailed_stats['total_images']
            }
        try:
            coll_size = int(collection.size().getInfo())
        except Exception as e:
            st.error(f"Failed to query collection size: {e}")
            coll_size = 0

        if coll_size == 0:
            st.warning("No images found for the selected criteria, increase cloud cover threshold,  change the date range, or make sure the acquisition date aligned with Landsat Mission Avaliability.")

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
    - **Available Date Range:** {detailed_stats.get('date_range', 'N/A')}
    """

    st.markdown(summary_md)
    # Path/Row information in expandable section
    path_row_tiles = detailed_stats.get('path_row_tiles', [])
    if path_row_tiles:
        with st.expander(f"WRS Path/Row Coverage ({len(path_row_tiles)} tiles)"):
            # Create columns for better display
            num_cols = 3
            cols = st.columns(num_cols)
            
            for idx, (path, row) in enumerate(path_row_tiles):
                col_idx = idx % num_cols
                cols[col_idx].write(f" Path {path:03d} / Row {row:03d}")

    #Detailed Scene Information with cloud cover information
    with st.expander("Scene ID, acquisition date, and cloud cover"):
        scene_ids = detailed_stats.get('Scene_ids', [])
        acquisition_dates = detailed_stats.get('individual_dates', [])
        cloud_covers = detailed_stats.get('cloud_cover', {}).get('values', [])
        
        if scene_ids and acquisition_dates:
            import pandas as pd
            
            # Create a dataframe with all information
            scene_df = pd.DataFrame({
                '#': range(1, len(scene_ids) + 1),
                'Scene ID': scene_ids,
                'Acquisition Date': acquisition_dates,
                'Cloud Cover (%)': [round(cc, 2) for cc in cloud_covers] if cloud_covers else ['N/A'] * len(scene_ids)
            })
            
            # Display the table with formatting
            st.dataframe(
                scene_df,
                width='stretch',
                hide_index=True,
                column_config={
                    '#': st.column_config.NumberColumn('#', width='small'),
                    'Scene ID': st.column_config.TextColumn('Scene ID', width='large'),
                    'Acquisition Date': st.column_config.TextColumn('Acquisition Date', width='medium'),
                    'Cloud Cover (%)': st.column_config.NumberColumn(
                        'Cloud Cover (%)',
                        width='medium',
                        format="%.2f"
                    )
                }
            )
            
            # Show cloud cover statistics
            if cloud_covers:
                st.markdown("#### ☁️ Cloud Cover Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Minimum", f"{min(cloud_covers):.2f}%")
                with col2:
                    st.metric("Average", f"{sum(cloud_covers)/len(cloud_covers):.2f}%")
                with col3:
                    st.metric("Maximum", f"{max(cloud_covers):.2f}%")
            
            # Download button
            csv = scene_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Scene List as CSV",
                data=csv,
                file_name=f"landsat_scenes_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
        else:
            st.info("No scene data available to display")
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

#export the data
if st.session_state.composite is not None and st.session_state.aoi is not None:
    st.subheader("Export Mosaic to Google Drive")
    
    with st.expander("Export Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # File naming
            default_name = f"Landsat_{st.session_state.search_metadata.get('sensor', 'unknown')}_{st.session_state.search_metadata.get('start_date', '')}_{st.session_state.search_metadata.get('end_date', '')}_mosaic"
            export_name = st.text_input(
                "Export Filename:",
                value=default_name,
                help="The output will be saved as GeoTIFF (.tif)"
            )
            
            drive_folder = st.text_input(
                "Google Drive Folder:",
                value="EarthEngine_Exports",
                help="Google Drive folder to store the result"
            )
            
        with col2:
            crs_options = {
                "WGS 84 (EPSG:4326)": "EPSG:4326",
                "Custom EPSG": "CUSTOM"
            }
            
            crs_choice = st.selectbox(
                "Coordinate Reference System:",
                options=list(crs_options.keys()),
                index=0
            )
            
            if crs_choice == 'Custom EPSG':
                custom_epsg = st.text_input(
                    "Enter EPSG Code:",
                    value="4326",
                    help="Example: 32648 (UTM Zone 48N)"
                )
                export_crs = f"EPSG:{custom_epsg}"
            else:
                export_crs = crs_options[crs_choice]
            
            scale = st.number_input(
                "Pixel Size (meters):",
                min_value=10,
                max_value=1000,
                value=30,
                step=10
            )
        
        if st.button("Start Export to Google Drive", type="primary"):
            try:
                with st.spinner("Preparing export task..."):
                    # Use the composite from session state
                    export_image = st.session_state.composite
                    
                    # Ensure we have valid band names
                    band_names = export_image.bandNames()
                    export_image = export_image.select(band_names)
                    
                    # Get the geometry from AOI - FIXED
                    aoi_obj = st.session_state.aoi
                    
                    # Convert to geometry based on type
                    if isinstance(aoi_obj, ee.FeatureCollection):
                        export_region = aoi_obj.geometry()
                    elif isinstance(aoi_obj, ee.Feature):
                        export_region = aoi_obj.geometry()
                    elif isinstance(aoi_obj, ee.Geometry):
                        export_region = aoi_obj
                    else:
                        # If all else fails, try to get bounds
                        try:
                            export_region = aoi_obj.geometry()
                        except:
                            raise ValueError(f"Cannot extract geometry from AOI object of type: {type(aoi_obj)}")
                    
                    # Set export parameters
                    export_params = {
                        "image": export_image,
                        "description": export_name.replace(" ", "_"),  # Remove spaces from description
                        "folder": drive_folder,
                        "fileNamePrefix": export_name,
                        "scale": scale,
                        "crs": export_crs,
                        "maxPixels": 1e13,
                        "fileFormat": "GeoTIFF",
                        "formatOptions": {"cloudOptimized": True},
                        "region": export_region
                    }
                    
                    # Start the export task
                    task = ee.batch.Export.image.toDrive(**export_params)
                    task.start()
                    
                    st.success(f"✅ Export task '{export_name}' submitted successfully!")
                    st.info(f"Task ID: {task.id}")
                    st.markdown(f"""
                    **Export Details:**
                    - File location: Google Drive/{drive_folder}/{export_name}.tif
                    - CRS: {export_crs}
                    - Resolution: {scale}m
                    
                    Check progress in the [Earth Engine Task Manager](https://code.earthengine.google.com/tasks)
                    """)
                    
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
                st.info("Debugging info:")
                st.write(f"AOI type: {type(st.session_state.aoi)}")
                st.write(f"Composite exists: {st.session_state.composite is not None}")

# Navigation
st.divider()
st.subheader("Module Navigation")

if st.session_state.composite is not None:
    if st.button("Go to Module 2: Classification Scheme"):
        st.switch_page("pages/2_Module_2_Classification_scheme.py")
else:
    st.button("🔒 Complete Module 1 First", disabled=True)
    st.info("Please generate an imagery mosaic before proceeding")