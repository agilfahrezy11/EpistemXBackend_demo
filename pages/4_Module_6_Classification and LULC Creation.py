import streamlit as st
import geemap.foliumap as geemap
from src.src_modul_6 import FeatureExtraction, Generate_LULC
import ee

#Page configuration
st.set_page_config(
    page_title="Land Cover Land Use Classification",
    layout="wide"
)

#Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    st.error(f"Earth Engine initialization failed: {e}")
    st.stop()
#Set the page title (for the canvas)
st.title("Land Cover Land Use Classification")
st.divider()
st.markdown("""
This module performs land cover land use classification using Random Forest classifier. Random Forest is a non-parametric machine learning classifiers widely used in remote sensing community.
            In order to use this module, you must complete module 1 - 4. Module 1 generates satellite imagery mosaic which serves as predictor variable used to generate the final classification. 
            Module 2 allows you to define the classification scheme which aims to list the class of the final map. Module 3 define the ROI or training data which used to trained the model 
            Module 4 allows you to analyze the quality of the training data using separability analysis and various plots
""")

#Sidebar
st.sidebar.title("About")
st.sidebar.info("Perform land cover classification using Google Earth Engine Random Forest classifier")
logo = "logos\logo_epistem.png"
st.sidebar.image(logo)

# Check prerequisites from previous modules. The module cannot open if the previous modules is not complete.
st.subheader("Prerequisites Check")

col1, col2 = st.columns(2)

# Check for image composite from Module 1
with col1:
    if 'composite' in st.session_state and st.session_state.composite is not None:
        st.success("✅ Image Composite Available (Module 1)")
        image = st.session_state['composite']
        
        # Display metadata if available
        if 'Image_metadata' in st.session_state:
            metadata = st.session_state['Image_metadata']
            with st.expander("Image Details"):
                st.write(f"**Sensor:** {st.session_state.get('search_metadata', {}).get('sensor', 'N/A')}")
                st.write(f"**Date Range:** {metadata.get('date_range', 'N/A')}")
                st.write(f"**Total Images:** {metadata.get('total_images', 'N/A')}")
    else:
        st.error("❌ Image Composite Not Found")
        st.warning("Please complete Module 1 first to generate an image composite")
        image = None

# Check for training data from Module 3/4
with col2:
    if 'training_data' in st.session_state and st.session_state.training_data is not None:
        st.success("✅ Training Data Available (Module 3)")
        roi = st.session_state['training_data']
        
        # Display training data info if available
        if 'training_gdf' in st.session_state:
            gdf = st.session_state['training_gdf']
            with st.expander("Training Data Details"):
                st.write(f"**Total Features:** {len(gdf)}")
                st.write(f"**Columns:** {', '.join(gdf.columns.tolist())}")
                
                # Show class distribution if class property is known
                if 'selected_class_property' in st.session_state:
                    class_prop = st.session_state['selected_class_property']
                    class_name = st.session_state['selected_class_name_property']
                    if class_prop in gdf.columns:
                        class_counts = gdf[class_prop].value_counts()
                        class_name = gdf[class_name].unique()
                        st.write("**Class Distribution:**")
                        st.dataframe(class_counts, width='stretch')
    else:
        st.error("❌ Training Data Not Found")
        st.warning("Please complete Module 3 to and analyze the training data")
        roi = None

# Stop if prerequisites are not met
if image is None or roi is None:
    st.divider()
    st.info("⚠️ Please complete the previous modules before proceeding with classification")
    st.markdown("""
    **Required Steps:**
    1. **Module 1:** Generate image composite
    2. **Module 3:** Upload and analyze training data (ROI)
    3. **Module 4:** Return here to perform classification
    """)
    st.stop()

# Get AOI for clipping
aoi = st.session_state.get('AOI', None)

# Initialize session state for storing results
if 'extracted_training_data' not in st.session_state:
    st.session_state.extracted_training_data = None
if 'extracted_testing_data' not in st.session_state:
    st.session_state.extracted_testing_data = None
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None

st.divider()

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["Feature Extraction", "Classification", "Visualization"])

# ==================== TAB 1: FEATURE EXTRACTION ====================
with tab1:
    st.header("Feature Extraction Configuration")
    st.subheader("Extraction Parameters")
    st.markdown("Prior to classification, feature extraction is required to get the pixel value from the imagery for each class in the Region of Interest.")
        
        # Get class property from previous module if available
    default_class_prop = st.session_state.get('selected_class_property', 'class')
        
        # Class property name
    class_property = st.text_input(
            "Class Property Name",
            value=default_class_prop,
            help="Column name in your ROI containing the class labels"
        )
        
        # Pixel size
    pixel_size = st.number_input(
            "Pixel Size (meters)",
            min_value=1,
            max_value=1000,
            value=30,
            help="Spatial resolution for sampling"
        )
        
        # Tile scale
    tile_scale = st.number_input(
            "Tile Scale",
            min_value=1,
            max_value=32,
            value=16,
            help="Factor for aggregating tiles (higher = more memory efficient)"
        )
        
    
    st.markdown("---")
    # Extract Features button
    if st.button("Extract Features", type="primary", width='content'):
        with st.spinner("Extracting features from imagery..."):
            try:
                # Simply extract pixel values from all ROI data
                training_data = image.sampleRegions(
                    collection=roi,
                    properties=[class_property],
                    scale=pixel_size,
                    tileScale=tile_scale
                )
                
                st.session_state.extracted_training_data = training_data
                st.session_state.extracted_testing_data = None  # No testing data
                st.session_state.class_property = class_property
                st.metric("Total Training Samples", training_data.size().getInfo())
                st.success("✅ Feature extraction completed!")
                
                st.info("ℹ️ All ROI data has been extracted. Make sure you prepared the validation data for accuracy assessment.")
                
            except Exception as e:
                st.error(f"Error during feature extraction: {e}")
                import traceback
                st.code(traceback.format_exc())


# ==================== TAB 2: CLASSIFICATION ====================
with tab2:
    st.header("Classification Configuration")
    
    # Check if training data is available
    if st.session_state.extracted_training_data is None:
        st.warning("Please extract features first in the 'Feature Extraction' tab")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Classification Method")
            
            # Classification mode selection
            classification_mode = st.radio(
                "Select Classification Mode",
                ["Hard Classification (Multiclass)", "Soft Classification (One-vs-Rest)"],
                help="Hard: Standard multiclass classification\nSoft: Probability-based with confidence layers"
            )
            
        with col2:
            st.subheader("Random Forest Hyperparameter")
            #Number of trees
            ntrees = st.number_input(
                "Number of Trees",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="More trees = better accuracy but slower computation"
            )
           #Variables per split

            use_auto_vsplit = st.checkbox(
                "Auto-calculate Variables Per Split",
                value=True,
                help="Automatically set to sqrt(number of bands)"
            )
            
            if not use_auto_vsplit:
                v_split = st.number_input(
                    "Variables Per Split",
                    min_value=1,
                    max_value=50,
                    value=5,
                    help="Number of variables to try at each split"
                )
            else:
                v_split = None
            # Minimum leaf population
            min_leaf = st.number_input(
                "Minimum Leaf Population",
                min_value=1,
                max_value=100,
                value=1,
                help="Minimum number of samples required in a leaf node"
            )
            
            # Soft classification specific parameters
            if classification_mode == "Soft Classification (One-vs-Rest)":
                st.markdown("---")
                st.subheader("Soft Classification Options")
                
                include_final_map = st.checkbox(
                    "Include Final Classification Map",
                    value=True,
                    help="Generate final map using argmax on probability layers"
                )
                
        
        st.markdown("---")
        
        # Classify button
        if st.button("Run Classification", type="primary", width='content'):
            with st.spinner("Running classification... This may take a few minutes."):
                try:
                    lulc = Generate_LULC()
                    
                    # Get the class property used during extraction
                    clf_class_property = st.session_state.get('class_property', class_property)
                    
                    if classification_mode == "Hard Classification (Multiclass)":
                        classification_result = lulc.hard_classification(
                            training_data=st.session_state.extracted_training_data,
                            class_property=clf_class_property,
                            image=image,
                            ntrees=ntrees,
                            v_split=v_split,
                            min_leaf=min_leaf,
                        )
                        st.session_state.classification_mode = "Hard Classification"
                    else:  # Soft Classification
                        classification_result = lulc.soft_classification(
                            training_data=st.session_state.extracted_training_data,
                            class_property=clf_class_property,
                            image=image,
                            include_final_map=include_final_map,
                            ntrees=ntrees,
                            v_split=v_split,
                            min_leaf=min_leaf,
                        )
                        st.session_state.classification_mode = "Soft Classification"
                        st.session_state.include_final_map = include_final_map
                    
                    st.session_state.classification_result = classification_result
                    st.session_state.classification_params = {
                        'mode': classification_mode,
                        'ntrees': ntrees,
                        'v_split': v_split,
                        'min_leaf': min_leaf,
                    }
                    
                    st.success("✅ Classification completed successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error during classification: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# ==================== TAB 3: RESULTS====================
with tab3:
    st.header("Classification Results")
    
    if st.session_state.classification_result is None:
        st.info("ℹ️ No classification results yet. Please run classification first.")
    else:
        st.success("✅ Classification completed!")
        
        # Summary section
        st.subheader("Classification Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mode = st.session_state.get('classification_mode', 'N/A')
            st.metric("Classification Mode", mode)
        
        with col2:
            params = st.session_state.get('classification_params', {})
            st.metric("Number of Trees", params.get('ntrees', 'N/A'))
        
        with col3:
            if st.session_state.extracted_training_data:
                train_size = st.session_state.extracted_training_data.size().getInfo()
                st.metric("Training Samples", train_size)
        
        with col4:
            if st.session_state.extracted_testing_data:
                test_size = st.session_state.extracted_testing_data.size().getInfo()
                st.metric("Testing Samples", test_size)
            else:
                st.metric("Testing Samples", "N/A")
        
        st.markdown("---")
        
        # Visualization section
        st.subheader("Classification Map Preview")
        
        if st.checkbox("Show Classification Map", value=True):
            try:
                # Prepare visualization
                classification_to_show = st.session_state.classification_result
                
                # If soft classification, show the final classification band
                if st.session_state.get('classification_mode') == "Soft Classification":
                    if st.session_state.get('include_final_map', False):
                        # Select the classification band
                        band_names = classification_to_show.bandNames().getInfo()
                        if 'classification' in band_names:
                            classification_to_show = classification_to_show.select('classification')
                        else:
                            st.warning("No final classification map found. Showing first probability band.")
                            classification_to_show = classification_to_show.select(0)
                
                # Create color palette based on number of classes
                if 'training_gdf' in st.session_state and 'selected_class_property' in st.session_state:
                    class_prop = st.session_state['selected_class_property']
                    gdf = st.session_state['training_gdf']
                    unique_classes = sorted(gdf[class_prop].unique())
                    
                    # Generate color palette
                    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
                             '#00FFFF', '#800000', '#008000', '#000080', '#808000']
                    palette = colors[:len(unique_classes)]
                    
                    vis_params = {
                        'min': min(unique_classes),
                        'max': max(unique_classes),
                        'palette': palette
                    }
                else:
                    vis_params = {
                        'min': 1,
                        'max': 10,
                        'palette': ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']
                    }
                
                # Create map
                if 'training_gdf' in st.session_state:
                    gdf = st.session_state['training_gdf']
                    centroid = gdf.geometry.centroid.iloc[0]
                    Map = geemap.Map(center=[centroid.y, centroid.x], zoom=10)
                else:
                    Map = geemap.Map()
                
                # Add layers
                Map.addLayer(classification_to_show, vis_params, 'Classification Result', True)
                Map.addLayer(image, st.session_state.get('visualization', {}), 'Image Composite', False)
                
                if 'training_gdf' in st.session_state:
                    Map.add_geojson(st.session_state['training_gdf'].__geo_interface__, 
                                   layer_name="Training Data", shown=False)
                
                Map.add_legend(title="Land Cover Classes", builtin_legend='NLCD')
                Map.to_streamlit(height=600)
                
            except Exception as e:
                st.error(f"Error displaying map: {e}")
                st.code(traceback.format_exc())
        
        st.markdown("---")

# Footer with navigation
st.divider()
st.subheader("Module Navigation")

col1, col2 = st.columns(2)

with col1:
    if st.button("⬅️ Back to Module 3: Analyze ROI", width='stretch'):
        st.switch_page("pages/3_Module_4_Analyze_ROI.py")

with col2:
    if st.session_state.classification_result is not None:
        if st.button("➡️ Go to Module 5: Accuracy Assessment", width='stretch'):
            st.info("Accuracy assessment module coming soon!")
    else:
        st.button("🔒 Complete Classification First", disabled=True, width='stretch')

# Show completion status
if st.session_state.classification_result is not None:
    st.success(f"✅ Classification completed using {st.session_state.get('classification_mode', 'N/A')}")
else:
    st.info("💡 Complete feature extraction and classification to proceed")