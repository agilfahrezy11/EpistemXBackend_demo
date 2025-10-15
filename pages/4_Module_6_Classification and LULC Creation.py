import streamlit as st
import pandas as pd
import plotly.express as px
import geemap.foliumap as geemap
from src.src_modul_6 import FeatureExtraction, Generate_LULC
import ee
import traceback

#Page configuration
st.set_page_config(
    page_title="Supervised Classification",
    page_icon="logos\logo_epistem_crop.png",
    layout="wide"
)
#Set the page title (for the canvas)
st.title("Generate Land Cover Map using Supervised Classification")
st.divider()
st.markdown("""
This module performs land cover land use classification using Random Forest classifier. Random Forest is a non-parametric machine learning classifiers widely used in remote sensing community.
            In order to use this module, you must complete module 1 - 4. Module 1 generates satellite imagery mosaic which serves as predictor variable used to generate the final classification. 
            Module 2 allows you to define the classification scheme which aims to list the class of the final map. Module 3 define the ROI or training data which used to trained the model 
            Module 4 allows you to analyze the quality of the training data using separability analysis and various plots
""")

#Sidebar info
st.sidebar.title("About")
st.sidebar.info("Module for generating a classification map based on Statistical Machine Intellegence and Learning (SMILE) Random Forest classifier")
logo = "logos\logo_epistem.png"
st.sidebar.image(logo)

# Check prerequisites from previous modules. The module cannot open if the previous modules is not complete.
st.subheader("Prerequisites Check")

col1, col2 = st.columns(2)

#Check for image composite from Module 1
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
        #Display error message if composite is not found
        st.error("❌ Image Composite Not Found")
        st.warning("Please complete Module 1 first to generate an image composite")
        image = None

#Check for training data from Module 3/4
with col2:
    if 'training_data' in st.session_state and st.session_state.training_data is not None:
        st.success("✅ Training Data Available")
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
        st.warning("Please complete Module 3 and 4 to create and analyze the Region of Interest (ROI)")
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

#Get AOI for clipping the result
aoi = st.session_state.get('AOI', None)


#Initialize session state for storing results
if 'extracted_training_data' not in st.session_state:
    st.session_state.extracted_training_data = None
if 'extracted_testing_data' not in st.session_state:
    st.session_state.extracted_testing_data = None
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None

st.divider()

#Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Feature Extraction", "Model Training", "Model Report", "Model Evaluation", "Visualization"])
#write each the content for each tab

# ==================== Tab 1: Feature Extraction ====================
#Option to either use all of the training data for classification, or split them into train and test data
with tab1:
    st.header("Feature Extraction Configuration")
    markdown = """ 
    The first step in the classification is extracting the pixel values of the imagery data for each class ROI. Prior to extracting the pixel, you must specified if you are going to split the ROI into training and testing data.
    If you decide to split the data, you will be able to evaluate the model of the classification, prior to generating the land cover classification.
    If you decide not to split the data, you cannot evaluate the model quality, and thus only able to compute thematic accuracy in module 7. 
    """
    st.markdown(markdown)
    
    col1, col2 = st.columns([1, 1])
    #first column, provide option to split or not split
    with col1:
        st.subheader("Data Split Options")
        # Option to split data
        split_data = st.checkbox(
            "Split data into training and testing subsets",
            value=True,
            help="If unchecked, all ROI data will be used for training the classifier"
        )
        #What happened if the user choose to split the data
        if split_data:
            st.info("The ROI is split into training and testing data using stratified random split approach")
            
            #Split ratio
            split_ratio = st.slider(
                "Training Data Ratio",
                min_value=0.5,
                max_value=0.9,
                value=0.7,
                step=0.05,
                help="Proportion of data to use for training"
            )
            #information about the proportion
            st.metric("Training", f"{split_ratio*100:.0f}%", delta=None)
            st.metric("Testing", f"{(1-split_ratio)*100:.0f}%", delta=None)
        #What happened if the user choose not to split the data    
        else:
            st.warning("All ROI data will be used for training. Please prepared an independent testing dataset.")
    #Second column, prepared the extraction parameters 
    with col2:
        st.subheader("Extraction Parameters")
        # Get class property from previous module if available. What the user choose for separability analysis, will be used here
        default_class_prop = st.session_state.get('selected_class_property', 'class')
        # Class property name
        class_property = st.text_input(
            "Class ID",
            value=default_class_prop,
            help="Column name in your ROI containing the class ID"
        )
        # Pixel size
        pixel_size = st.number_input(
            "Pixel Size (meters)",
            min_value=1,
            max_value=1000,
            value=30,
            help="Spatial resolution for sampling"
        )
    st.markdown("---")
    
    #Extract Features button
    if st.button("Extract Features", type="primary", width='stretch'):
        #Spinner to show progress
        with st.spinner("Extracting features from imagery..."):
            try:
                #Use module 6 feature extraction class 
                fe = FeatureExtraction()
                #define the spliting function from the source code
                if split_data:
                    # Use Stratified Random Split
                    training_data, testing_data = fe.stratified_split(
                        roi=roi,
                        image=image,
                        class_prop=class_property,
                        pixel_size=pixel_size,
                        train_ratio=split_ratio,
                    )
                    #stored the result in session state so that it can be used in classification
                    st.session_state.extracted_training_data = training_data
                    st.session_state.extracted_testing_data = testing_data
                    st.session_state.class_property = class_property
                    
                    st.success("✅ Feature extraction completed using Stratified Random Split!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Training Samples", training_data.size().getInfo())
                    with col2:
                        st.metric("Testing Samples", testing_data.size().getInfo())
                else:
                    #Extract all features without splitting
                    training_data = image.sampleRegions(
                        collection=roi,
                        properties=[class_property],
                        scale=pixel_size,
                    )
                    #store the data for the classification
                    st.session_state.extracted_training_data = training_data
                    st.session_state.extracted_testing_data = None
                    st.session_state.class_property = class_property
                    
                    st.success("✅ Feature extraction completed!")
                    st.info("ℹ️ All ROI data has been used for training. No test set created.")
            #error log if something fail    
            except Exception as e:
                st.error(f"Error during feature extraction: {e}")
                import traceback
                st.code(traceback.format_exc())

# ==================== Tab 2: Model Learning ====================
with tab2:
    st.header("Classification Configuration")
    st.markdown("The algoritm use for conducting the classification is Random Forest. You need to specified the value of three main Random Forest parameters. This parameters are:")
    st.markdown("1. Number of Trees: Control the number of decision tree in the model. Ideal value vary, but most remote sensing application utilize >300 tree")
    st.markdown("2. Variables Per Split: Number of variable selected for conducting a split. You can also used the default value, which is the square root of the number of variables used")
    st.markdown("3. Minimum Leaf Population: Number of minimum sample selected for splitting a leaf node")
    
    #Check if training data is available
    if st.session_state.extracted_training_data is None:
        st.warning("Please extract features first in the 'Feature Extraction' tab")
    else:
        col1, col2 = st.columns([1, 1])
        #First column, choosing hard or soft classification (could be remove later)
        with col1:
            st.subheader("Classification Approach")
            
            # Classification mode selection
            classification_mode = st.radio(
                "Select Classification Mode",
                ["Hard Classification (Multiclass)", "Soft Classification (One-vs-Rest)"],
                help="Hard: Standard multiclass classification\nSoft: Probability-based with confidence layers"
            )
        #column for hyperparameter value
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
           #default value of variable per split, the sqrt of number of bands 
            use_auto_vsplit = st.checkbox(
                "Use default value of Variables Per Split",
                value=True,
                help="Automatically set to sqrt(number of bands)"
            )
            #User can define their own variable per split value
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
            with st.spinner("Running classification...."):
                try:
                    #Initialze the generate lulc class from module 6
                    lulc = Generate_LULC()
                    #Get the class property used during extraction. Its define as parameters since it will be used again for visualization
                    clf_class_property = st.session_state.get('class_property', class_property)
                    #Run the module 6 LULC hard classification
                    if classification_mode == "Hard Classification (Multiclass)":
                        classification_result, trained_model = lulc.hard_classification(
                            training_data=st.session_state.extracted_training_data,
                            class_property=clf_class_property,
                            image=image,
                            ntrees=ntrees,
                            v_split=v_split,
                            min_leaf=min_leaf,
                            return_model=True
                        )
                        #Store the result for visualization
                        st.session_state.classification_mode = "Hard Classification"
                        st.session_state.trained_model = trained_model
                        st.session_state.classification_result = classification_result
                    #Soft Classification
                    #At this point, the soft classification is not modified, since in order to evaluate them, it required cross entropy loss metric
                    else:  
                        classification_result = lulc.soft_classification(
                            training_data=st.session_state.extracted_training_data,
                            class_property=clf_class_property,
                            image=image,
                            include_final_map=include_final_map,
                            ntrees=ntrees,
                            v_split=v_split,
                            min_leaf=min_leaf,
                        )
                        #Store the result for visualization
                        st.session_state.classification_mode = "Soft Classification"
                        st.session_state.include_final_map = include_final_map
                    
                    st.session_state.classification_result = classification_result
                    st.session_state.classification_params = {
                        'mode': classification_mode,
                        'ntrees': ntrees,
                        'v_split': v_split,
                        'min_leaf': min_leaf,
                        'class_property': clf_class_property

                    }
                    
                    st.success("✅ Classification completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error during classification: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# ==================== TAB 3 Summary Result ====================
#Main concern. user still able to get summary report without the model accuaracy
with tab3:
    st.header("Classification Summary and Report")
    st.markdown("This section shows the model performance on the test dataset. Using this appproach, we can evaluate how well the Random Forest model" \
    "learned from the training data")
    #Check the avaliability classification model
    if st.session_state.classification_result is None:
        st.warning("Complete the classification to evaluates the performance")
        st.stop()
    #Check the trained model, if not avaliable do not run
    if 'trained_model' not in st.session_state:
        st.error("Trained model is not found. Please re-run classification.")
        st.stop()
    #If avaliable 
    else:
        st.success("Testing data avaliable for model accuracy")
    # ==== Model Information =====
    st.subheader("Model configuration")
    #Get the classification parameter
    params = st.session_state.get('classification_params', {})
    col1, col2, col3 = st.columns(3)
    #column 1 for decision tree
    with col1:
        st.metric("Number of Decision Tree", params.get('ntrees', 'N/A'))
    #column 2 for variable split
    with col2:
        v_split = params.get('v_split', 'Default')
        st.metric("Variable selected at split", v_split if v_split else 'Default')
    #column 3 for minimum leaf population
    with col3:
        st.metric("minimum leaf population", params.get('min_leaf', 'N/A'))
    # ==== Feature Importance ====
    st.subheader("Feature Importance Analysis")
    #Feature importance analysis located source code of module 6
    try:
        lulc = Generate_LULC()
        #Get the feature importance using the source code
        importance_df = lulc.get_feature_importance(
            st.session_state.trained_model
        )
        #Store for later use
        st.session_state.importance_df = importance_df
        #visualize the feature importance
        col1, col2 = st.columns([2, 1])
        #first collumn, display as bar chart
        with col1:
            #Use the plotly for interactive visualization
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Band',
                orientation='h',
                title='Variable Importance Ranking',
                color='Importance',
                color_continuous_scale='Viridis',
                text='Importance (%)'
            )
            
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=max(400, len(importance_df) * 30),
                showlegend=False
            )
            
            st.plotly_chart(fig, width = 'stretch')
        #Display the most importance features
        with col2:
            st.markdown("**Top 5 Most Important Features:**")
            for i, row in importance_df.head(5).iterrows():
                st.metric(
                    f"{i+1}. {row['Band']}", 
                    f"{row['Importance (%)']:.1f}%"
                )
            
            # Show full table
            with st.expander("Complete Importance Table"):
                st.dataframe(
                    importance_df.style.background_gradient(
                        subset=['Importance (%)'],
                        cmap='YlGn'
                    ),
                    width='stretch',
                    hide_index=True
                )

    except Exception as e:
        st.error(f"Error retrieving feature importance: {e}")
    
# ==================== TAB 4 Model Evaluation ====================
with tab4:
    st.header("Model Evaluation")
        #check the testing data
    have_test_data = st.session_state.extracted_testing_data is not None
    if not have_test_data:
        st.warning(" No testing data available. Cannot evaluate the model, use the freature extraction tab to split the ROI into training and testing data ")
        st.info("""
        **To perform Model evaluation:**
        1. Go to the 'Feature Extraction' tab
        2. Enable 'Split data into Training and Testing sets'
        3. Re-run feature extraction and classification
        4. Return here to evaluate the model
        """)
        #show model information without its accuracy
    else:
        # Button to compute accuracy
        if st.button("Evaluate model accuracy", type="primary", width='content'):
            with st.spinner("Evaluating the model..."):
                try:
                    lulc = Generate_LULC()
                    class_prop = st.session_state.get('classification_params', {}).get('class_property')
                    #st.session_state['selected_class_property']
                    #Use the functions in the source code to perform model evaluation
                    model_quality = lulc.evaluate_model(
                        trained_model=st.session_state.trained_model,
                        test_data=st.session_state.extracted_testing_data,
                        class_property=class_prop
                    )
                    #Store in session state
                    st.session_state.model_quality = model_quality
                    
                    st.success("✅ Model Evaluation Complete!")
                except Exception as e:
                    st.error(f"Error during model evaluation: {e}")
                    st.code(traceback.format_exc())
            #Shows the result if complete
        if "model_quality" in st.session_state:
            st.subheader("Model Accuracy Report")
            acc = st.session_state.model_quality
            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Accuracy", f"{acc['overall_accuracy']*100:.2f}%")
            col2.metric("Kappa Coefficient", f"{acc['kappa']:.3f}")
            mean_f1 = sum(acc['f1_scores']) / len(acc['f1_scores'])
            col3.metric("Mean F1-score", f"{mean_f1:.3f}")

            st.markdown("---")
            st.subheader("Class-level Metrics")

            #Convert Producer (Recall) and Consumer (Precision) Accuracies into a DataFrame
            df_metrics = pd.DataFrame({
                "Class ID": range(len(acc["precision"])),
                "Producer's Accuracy (Recall)": acc["precision"],
                "User's Accuracy (Precision)": acc["recall"],
                "F1-score": acc["f1_scores"]
            })
            df_metrics["Producer's Accuracy (Recall)"] = (df_metrics["Producer's Accuracy (Recall)"] * 100).round(2)
            df_metrics["User's Accuracy (Precision)"] = (df_metrics["User's Accuracy (Precision)"] * 100).round(2)
            df_metrics["F1-score"] = df_metrics["F1-score"].round(3)

            st.dataframe(df_metrics, width='stretch')

            # Plot Confusion Matrix as heatmap
            st.subheader("Confusion Matrix")
            cm = pd.DataFrame(
                acc["confusion_matrix"],
                columns=[f"Pred_{i}" for i in range(len(acc["confusion_matrix"]))],
                index=[f"Actual_{i}" for i in range(len(acc["confusion_matrix"]))]
            )

            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                title="Confusion Matrix"
            )
            st.plotly_chart(fig, width='stretch')

# ==================== TAB 5 Visualization ====================
with tab5:
    st.header("Classification Report")
    
    if st.session_state.classification_result is None:
        st.info("ℹ️ No classification results yet. Please run classification first.")
    else:
        st.success("✅ Classification completed!")
        # Visualization section
        st.subheader("Classification Map Preview")
        if st.checkbox("Show Classification Map", value=True):
            try:
                # Prepare visualization
                classiifcation_map = st.session_state.classification_result
                
                # If soft classification, show the final classification band
                if st.session_state.get('classification_mode') == "Soft Classification":
                    if st.session_state.get('include_final_map', False):
                        # Select the classification band
                        band_names = classiifcation_map.bandNames().getInfo()
                        if 'classification' in band_names:
                            classiifcation_map = classiifcation_map.select('classification')
                        else:
                            st.warning("No final classification map found. Showing first probability band.")
                            classiifcation_map = classiifcation_map.select(0)
                
                # Create color palette based on number of classes
                # Create custom color palette with user input
                if 'training_gdf' in st.session_state and 'selected_class_property' in st.session_state:
                    class_prop = st.session_state['selected_class_property']
                    class_name_prop = st.session_state.get('selected_class_name_property', None)
                    gdf = st.session_state['training_gdf']
                    unique_classes = sorted(gdf[class_prop].unique())
                    
                    # Allow user to customize colors
                    st.subheader("Customize Map Colors")
                    
                    with st.expander("Define Class Colors", expanded=False):
                        st.markdown("Assign colors to each land cover class:")
                        
                        # Create color mapping dictionary
                        if 'class_colors' not in st.session_state:
                            # Initialize with default colors
                            default_colors = ['#228B22', '#0000FF', '#FF0000', '#FFFF00', '#8B4513', 
                                            '#808080', '#FFA500', '#00FFFF', '#FF00FF', '#90EE90']
                            st.session_state.class_colors = {
                                cls: default_colors[i % len(default_colors)] 
                                for i, cls in enumerate(unique_classes)
                            }
                        
                        # Create color pickers for each class
                        cols = st.columns(3)
                        for idx, class_id in enumerate(unique_classes):
                            with cols[idx % 3]:
                                # Get class name if available
                                if class_name_prop and class_name_prop in gdf.columns:
                                    class_name = gdf[gdf[class_prop] == class_id][class_name_prop].iloc[0]
                                    label = f"Class {class_id}: {class_name}"
                                else:
                                    label = f"Class {class_id}"
                                
                                # Color picker
                                st.session_state.class_colors[class_id] = st.color_picker(
                                    label,
                                    value=st.session_state.class_colors.get(class_id, '#228B22'),
                                    key=f"color_{class_id}"
                                )
                        
                        # Reset to default colors button
                        if st.button("🔄 Reset to Default Colors"):
                            default_colors = ['#228B22', '#0000FF', '#FF0000', '#FFFF00', '#8B4513', 
                                            '#808080', '#FFA500', '#00FFFF', '#FF00FF', '#90EE90']
                            st.session_state.class_colors = {
                                cls: default_colors[i % len(default_colors)] 
                                for i, cls in enumerate(unique_classes)
                            }
                            st.rerun()
                    
                    # Build palette from user selections
                    palette = [st.session_state.class_colors[cls] for cls in unique_classes]
                    
                    vis_params = {
                        'min': min(unique_classes),
                        'max': max(unique_classes),
                        'palette': palette
                    }
                
                # Create map
                if 'training_gdf' in st.session_state:
                    gdf = st.session_state['training_gdf']
                    centroid = gdf.geometry.centroid.iloc[0]
                    Map = geemap.Map(center=[centroid.y, centroid.x], zoom=10)
                else:
                    Map = geemap.Map()
                
                # Add layers
                Map.addLayer(classiifcation_map, vis_params, 'Random Forest Classification', True)
                Map.addLayer(image, st.session_state.get('visualization', {}), 'Image Composite', False)
                
                #if 'training_gdf' in st.session_state:
                #    Map.add_geojson(st.session_state['training_gdf'].__geo_interface__, 
                #                   layer_name="Training Data", shown=False)
    
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
        if st.button("➡️ Go to Module 7: Thematic Accuracy Assessment", width='stretch'):
            st.info("Accuracy assessment module coming soon!")
    else:
        st.button("🔒 Complete Classification First", disabled=True, width='stretch')

# Show completion status
if st.session_state.classification_result is not None:
    st.success(f"Classification completed using {st.session_state.get('classification_mode', 'N/A')}")
else:
    st.info("💡 Complete feature extraction and classification to proceed")