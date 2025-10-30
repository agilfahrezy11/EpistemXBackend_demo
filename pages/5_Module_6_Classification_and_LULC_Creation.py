import streamlit as st
import pandas as pd
import plotly.express as px
import geemap.foliumap as geemap
from epistemx.module_6_phase1 import FeatureExtraction, Generate_LULC
import numpy as np
import traceback

#Page configuration
st.set_page_config(
    page_title="Supervised Classification",
    page_icon="logos\logo_epistem_crop.png",
    layout="wide"
)
#Set the page title (for the canvas)
st.title("Land Cover Map Generations")
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

#Check prerequisites from previous modules. The module cannot open if the previous modules is not complete.
#add module 2 check and module 3 (for training data not analysis)
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

#Stop if prerequisites are not met
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
tab1, tab2, tab3, tab4 = st.tabs(["Feature Extraction", "Model Training", "Model Summary and Evaluation", "Visualization"])
#write each the content for each tab

# ==================== Tab 1: Feature Extraction ====================
#Option to either use all of the training data for classification, or split them into train and test data
#This section can be change to module 3 (?)
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
        #Get class property from previous module if available. What the user choose for separability analysis, will be used here
        default_class_prop = st.session_state.get('selected_class_property', 'class')
        #Class property name
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
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Training Samples", training_data.size().getInfo())
                    with col2:
                        st.metric("Testing Samples", testing_data.size().getInfo())
                    st.success("✅ Feature extraction complete!")
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
                    st.info("ℹ️ All ROI data has been used for training. No test set created.")
            #error log if something fail    
            except Exception as e:
                st.error(f"Error during feature extraction: {e}")
                import traceback
                st.code(traceback.format_exc())

# ==================== Tab 2: Model Learning ====================
with tab2:
    st.header("Membuat Model Klasifikasi")
    
    #introduction
    st.markdown("""
    Di bagian ini, dilakukan proses klasifikasi digital untuk mengelompokkan pola penutup lahan pada citra satelit.
    Bayangkan anda menyuruh komputer untuk mengenali pola - pola, selayaknya anda melihat pola penutup lahan yang berbeda secara visual.
    """)
    
    #Algorithm explanation with visual
    with st.expander("🤔 Bagaimana Model Random Forest Mengenali Pola? (Click to learn more)", expanded=False):
        st.markdown("""
        **Random Forest:** Bayangkan model ini sebagai sekelompok ilmuwan ('pohon') 
        yang memberikan suara (voting) terkait jenis piksel pada citra satelit. 
        Proses pengelompokan nilai piksel menjadi kelas penutup lahan adalah sebagai berikut:
        
        🌲 **Setiap "Pohon"** mempertimbangkan kombinasi nilai piksel yang berbeda pada setiap kanal spektral
        🗳️ **Pengambilan Keputusan** Pohon ini kemudian menentukan tipe penutup lahan yang diwakili oleh setiap nilai piksel
        📊 **Keputusan Akhir** ditetapkan melalui pengambilan suara terbanyak, apapun yang disetujui oleh sebagian besar pohon akan menjadi keputusan terakhir
        
        **Fun fact: Random Forest menjadi salah satu algoritma yang banyak digunakan dalam kajian penginderaan jauh**
        - Dapat diandalkan karena proses penentuan kelas dilakukan melalui kumpulan 'pendapat ahli'
        - Dapat menghadapi berbagai jenis kondisi data (tidak seimbang, atau penuh dengan noise)
        """)
    
    #Check if training data is available
    if st.session_state.extracted_training_data is None:
        st.warning("⚠️ Lakukan ekstraksi nilai piksel melalui 'feature extraction'")
    else:
        st.success("✅ Proses ekstraksi nilai piksel tersedia. Proses klasifikasi dapat dilakukan")
        
        #Model Config with explanations
        st.subheader("Pengaturan Model Klasifikasi")
        with st.expander("Kenapa Model klasifikasi perlu diatur?", expanded = False):
            st.markdown(""" 
            Setiap model machine learning memiliki beberapa parameter yang mengendalikan bagaimana mesin
            mempelajari hubungan antara variabel dan pola data yang diberikan. Oleh karena itu, 
            pengaturean parameter ini dapat mempengaruhi kualitas model dan klasifikasi yang dihasilkan.
            
            """


            )

        st.markdown("Anda dapat memilih opsi untuk menggunakan pengaturan model yang telah disediakan atau mengatur dengan sendiri")
        
        #Create tabs for preset value, or manuall setting
        config_tab1, config_tab2 = st.tabs(["Setelan Umum", "⚙️ Pengaturan Lebih Lanjut"])
        #Preset parameter value
        with config_tab1:
            st.markdown("Pengaturan - pengaturan umumnya.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🌲 Number of Trees")
                st.markdown("*How many 'expert opinions' should we get?*")
                
                # Preset options for beginners
                tree_preset = st.radio(
                    "Choose a preset:",
                    ["Fast (50 trees) - Quick results", 
                     "Balanced (100 trees) - Good balance ⭐", 
                     "Accurate (200 trees) - Best results"],
                    index=1,
                    help="More trees = better accuracy but takes longer to run"
                )
                
                if "Fast" in tree_preset:
                    ntrees = 50
                elif "Balanced" in tree_preset:
                    ntrees = 100
                else:
                    ntrees = 200
                
                st.info(f"Using **{ntrees} trees** - {tree_preset.split(' - ')[1]}")
            
            with col2:
                st.markdown("### 🎯 Other Settings")
                st.markdown("*We'll use the best default values*")
                
                use_auto_vsplit = True
                v_split = None
                min_leaf = 1
                
                st.success("✅ **Variables per split:** Automatic (recommended)")
                st.success("✅ **Minimum samples:** 1 (standard)")
                st.info("💡 These defaults work great for most projects!")
        
        with config_tab2:
            st.markdown("**For users who want full control over the model parameters.**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🌲 Number of Trees")
                ntrees = st.number_input(
                    "Number of Trees",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                    help="More trees generally improve accuracy but increase computation time. 50-200 is usually sufficient."
                )
                
                # Visual feedback
                if ntrees < 50:
                    st.warning("⚡ Fast but may be less accurate")
                elif ntrees <= 150:
                    st.success("⚖️ Good balance of speed and accuracy")
                else:
                    st.info("🎯 High accuracy but slower processing")
            
            with col2:
                st.markdown("### 🔀 Variables per Split")
                use_auto_vsplit = st.checkbox(
                    "Use automatic selection (recommended)",
                    value=True,
                    help="Automatically chooses the optimal number based on your data"
                )
                
                if not use_auto_vsplit:
                    v_split = st.number_input(
                        "Variables Per Split",
                        min_value=1,
                        max_value=50,
                        value=5,
                        help="How many variables each tree considers at each split. Lower values add more randomness."
                    )
                else:
                    v_split = None
                    st.success("✅ Will use √(number of bands)")
            
            with col3:
                st.markdown("### 🍃 Minimum Leaf Size")
                min_leaf = st.number_input(
                    "Minimum Samples per Leaf",
                    min_value=1,
                    max_value=100,
                    value=1,
                    help="Minimum number of samples required in each leaf node. Higher values prevent overfitting."
                )
                
                if min_leaf == 1:
                    st.info("📊 Standard setting")
                elif min_leaf <= 5:
                    st.success("🛡️ Good for preventing overfitting")
                else:
                    st.warning("⚠️ May be too restrictive")
        
        # Ready to train section
        st.markdown("---")
        st.subheader("🚀 Ready to Train Your Model?")
        
        # Show current configuration summary
        with st.expander("📋 Current Configuration Summary", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🌲 Number of Trees", ntrees)
            with col2:
                if v_split is None:
                    st.metric("🔀 Variables per Split", "Auto")
                else:
                    st.metric("🔀 Variables per Split", v_split)
            with col3:
                st.metric("🍃 Min Samples per Leaf", min_leaf)
        
        # Estimated time warning
        if ntrees >= 200:
            st.warning("⏱️ **Heads up!** With 200+ trees, this might take a few minutes. Perfect time for a coffee break! ☕")
        elif ntrees >= 100:
            st.info("⏱️ **Estimated time:** 1-3 minutes depending on your data size.")
        else:
            st.success("⏱️ **Estimated time:** Less than 1 minute - nice and quick!")
        
        # The big classification button
        if st.button("🎯 Train Classification Model", type="primary", width='stretch'):
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Initializing Random Forest model...")
            progress_bar.progress(10)
            
            try:
                #Initialize the generate lulc class from module 6
                lulc = Generate_LULC()
                #Get the class property used during extraction
                clf_class_property = st.session_state.get('class_property')
                
                status_text.text(f"🌱 Training {ntrees} decision trees...")
                progress_bar.progress(30)
                
                #Run the hard classification
                classification_result, trained_model = lulc.hard_classification(
                    training_data=st.session_state.extracted_training_data,
                    class_property=clf_class_property,
                    image=image,
                    ntrees=ntrees,
                    v_split=v_split,
                    min_leaf=min_leaf,
                    return_model=True
                )
                
                status_text.text("💾 Saving results...")
                progress_bar.progress(80)
                
                #Store the results for visualization and evaluation
                st.session_state.classification_mode = "Hard Classification"
                st.session_state.trained_model = trained_model
                st.session_state.classification_result = classification_result
                st.session_state.classification_params = {
                    'mode': 'Hard Classification',
                    'ntrees': ntrees,
                    'v_split': v_split,
                    'min_leaf': min_leaf,
                    'class_property': clf_class_property
                }
                
                progress_bar.progress(100)
                status_text.text("🎉 Training completed!")
                
                # Success message with next steps
                st.success("🎉 **Congratulations!** Your classification model has been trained successfully!")
                st.info("👉 **What's next?** Go to the 'Model Summary and Evaluation' tab to see how well your model performed!")
                
                # Show a quick preview of what was accomplished
                st.markdown("### ✅ What we just accomplished:")
                st.markdown(f"- ✅ Trained a Random Forest model with **{ntrees} decision trees**")
                st.markdown(f"- ✅ Model learned to recognize **{len(st.session_state.get('lulc_classes_final', []))} different land cover types**")
                st.markdown("- ✅ Ready to classify your entire study area!")
                
            except Exception as e:
                progress_bar.progress(0)
                status_text.text("")
                st.error("❌ **Oops! Something went wrong during training.**")
                st.error("**Error details:** " + str(e))
                
                with st.expander("🔧 Technical Details (for troubleshooting)"):
                    import traceback
                    st.code(traceback.format_exc())
                
                st.markdown("### 💡 **Possible solutions:**")
                st.markdown("- Check that your training data has valid class labels")
                st.markdown("- Try reducing the number of trees if you're running out of memory")
                st.markdown("- Make sure your satellite imagery and training data overlap geographically")

# ==================== TAB 3 Summary Result ====================
#Main concern. user still able to get summary report without the model accuaracy
with tab3:
    st.header("Model Summary and Evaluation")
    st.markdown("This section shows model parameters, feature importance analysis, and accuracy of the model (if test data avaliable). This is one of the advantage of non-parametric machine learning classifier." \
    "It allows the analysis of the classification/model performance prior to generating a categorical data. The subsection of this tabs is as follows:")
    st.markdown("1. Model Parameters: Recap on the parameters used for the classification")
    st.markdown("2. Feature Importance Analysis: How each covariates benefits the model")
    st.markdown("3. Model Evaluation: Evaluate the accuracy of the model based on a seperate hold up test. This option is only avaliable if you decide to split the data in feature extraction tab")
    
    #Check the avaliability classification model
    if st.session_state.classification_result is None:
        st.warning("Complete the model learning tab to show the summary")
        st.stop()
    #Check the trained model, if not avaliable do not run
    if 'trained_model' not in st.session_state:
        st.error("Trained model is not found. Please re-run classification.")
        st.stop()
    #If avaliable 
    else:
        st.success("Testing data avaliable for model accuracy")
    st.divider()
    # ==== Model Information =====
    st.subheader("Model Parameters")
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
    st.divider()
    # ==== Feature Importance ====
    st.subheader("Feature Importance Analysis")
    st.markdown("Feature importance analysis indicate how each covariates benefits the classification." \
    "The higher value indicate higher importance which shows that model benefits from that covairates, and vice versa.")
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
                text='Importance'
            )
            
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=max(400, len(importance_df) * 30),
                showlegend=False
            )
            
            st.plotly_chart(fig,  use_container_width=True)
        #Display the most importance features
        with col2:
            st.markdown("**Top 5 Most Important Features:**")
            for i, row in importance_df.head(5).iterrows():
                st.write(f"{i+1}. {row['Band']}")
            
            # Show full table
            with st.expander("Complete Importance Table"):
                st.dataframe(
                    importance_df.style.background_gradient(
                        subset=['Importance'],
                        cmap='YlGn'
                    ),
                    width='stretch',
                    hide_index=True
                )
    except Exception as e:
        st.error(f"Error retrieving feature importance: {e}")
    st.divider()
    # ==== Model Evaluation ====
    st.subheader("Model Evaluation")
    st.markdown("This section allows you to evaluate the quality of the model based on the test data not used during the model learning. Therefore, quality of the model is tested on a subset of data that have not seen by the model itself." \
    " However, You can only do this if you split the data on feature extraction tab. If you decide not to split the data, you can only evaluate the thematic accuracy in module 7")
    st.markdown("Model evaluation procedure follows the same way as thematic accuracy assessment, using a confusion or matrix. Several accuracy metrics are used to evaluate the model. " \
    "Additionally, you can also evaluate the accuracy for each class, as well as inspect the resulting cofusion matrix")
    st.markdown("1. Overall Accuracy")
    st.markdown("2. Kappa Coefficient")
    st.markdown("3. Mean F1-score")
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
            #the metrics for overall model quality
            acc = st.session_state.model_quality
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall Accuracy", f"{acc['overall_accuracy']*100:.2f}%")
            col2.metric("Kappa Coefficient", f"{acc['kappa']:.3f}")
            mean_f1 = sum(acc['f1_scores']) / len(acc['f1_scores'])
            col3.metric("Mean F1-score", f"{mean_f1:.3f}")
            overall_gmean = acc.get('overall_gmean', 0)
            col4.metric("Overall G-Mean", f"{overall_gmean:.3f}")

            st.markdown("---")
            st.subheader("Class-level Metrics")

            #Convert Producer (Recall) and Consumer (Precision) Accuracies into a DataFrame
            # Build class-level metrics table in percentage form
            df_metrics = pd.DataFrame({
                "Class ID": range(len(acc["precision"])),
                "Producer's Accuracy (Recall) (%)": np.round(np.array(acc["recall"]) * 100, 2),
                "User's Accuracy (Precision) (%)": np.round(np.array(acc["precision"]) * 100, 2),
                "F1-score (%)": np.round(np.array(acc["f1_scores"]) * 100, 2),
                "Geometric Mean Score (%)": np.round(np.array(acc["gmean_per_class"]) * 100, 2)
            })

            st.dataframe(df_metrics, use_container_width=True)

            #Plot Confusion Matrix as heatmap
            st.subheader("Confusion Matrix")
            cm = pd.DataFrame(
                acc["confusion_matrix"],
                columns=[f"Pred_{i}" for i in range(len(acc["confusion_matrix"]))],
                index=[f"Actual_{i}" for i in range(len(acc["confusion_matrix"]))]
            )
            #Show the heatmap and customized it if needed
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                #title="Confusion Matrix"
            )
            fig.update_layout(
                autosize=True,
                height=600
            )
            st.plotly_chart(fig,     
                use_container_width=True, #got warning to upgrade to use 'use_container_width'
                config={
                    "displayModeBar": True,
                    "responsive": True
                 })

# ==================== TAB 4 Visualization ====================
with tab4:
    st.header("Visualization")
    
    if st.session_state.classification_result is None:
        st.info("ℹ️ No classification results yet. Please run classification first.")
    else:
        st.success("✅ Classification completed!")
        # Visualization section
        st.subheader("Classification Map Preview")
        if st.checkbox("Show Classification Map", value=True):
            try:
                # Prepare visualization
                classification_map = st.session_state.classification_result
                
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
                Map.addLayer(classification_map, vis_params, 'Random Forest Classification', True)
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
        st.switch_page("pages/4_Module_4_Analyze_ROI.py")

with col2:
    if st.session_state.classification_result is not None:
        if st.button("➡️ Go to Module 7: Thematic Accuracy Assessment", width='stretch'):
            st.switch_page("pages/6_Module_7_Thematic_Accuracy.py")
            st.info("Accuracy assessment module coming soon!")
    else:
        st.button("🔒 Complete Classification First", disabled=True, width='stretch')

# Show completion status
if st.session_state.classification_result is not None:
    st.success(f"Classification completed using {st.session_state.get('classification_mode', 'N/A')}")
else:
    st.info("💡 Complete feature extraction and classification to proceed")