"""
Module 6: Supervised Classification

This module facilitate the user to perform supervised classification using random forest classifier

Architecture:
- Backend (module_6_phase1.py): Pure backend process without UI dependencies
- Frontend (this file): Streamlit UI with session state management
- State synchronization ensures data persistence across page interactions
"""

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
    page_icon="logos/logo_epistem_crop.png",
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
logo = "logos/logo_epistem.png"
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
                        st.dataframe(class_counts, use_container_width=True)
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
    if st.button("Extract Features", type="primary", use_container_width=True):
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
        st.subheader("⚙️ Pengaturan Model Klasifikasi")
        with st.expander("Kenapa Model klasifikasi perlu diatur?", expanded = False):
            st.markdown(""" 
            Setiap model machine learning memiliki beberapa parameter yang mengendalikan bagaimana mesin
            mempelajari hubungan antara variabel dan pola data yang diberikan. Oleh karena itu, 
            pengaturean parameter ini dapat mempengaruhi kualitas model dan klasifikasi yang dihasilkan.
            """
            )
            st.markdown("Algoritma Random Forest memiliki beberapa parameter utama yang mempengaruhi kemampuannya untuk mempelajari pola")
            st.markdown("1. Jumlah Pohon Keputusan (number of trees)")
            st.markdown("2. Jumlah variabel yang dipertimbangkan saat pengambilan keputusan (variable_per_split)")
            st.markdown("3. Jumlah sampel yang dipertmbangkan untuk memecah sebuah daun dalam pohon keputusan (min leaf population)")
        st.markdown("Anda dapat memilih opsi untuk menggunakan pengaturan model yang telah disediakan atau mengatur dengan sendiri")
        
        #Create tabs for preset value, or manuall setting
        config_tab1, config_tab2 = st.tabs(["Pengaturan Umum", "⚙️ Pengaturan Lebih Lanjut"])
        #Preset parameter value
        with config_tab1:
            st.markdown("Pengaturan Umum yang diterapkan untuk kajian penginderaan jauh")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Jumlah Pohon Keputusan")
                st.markdown("*Berapa banyak 'pendapat ahli' diperlukan?*")
                
                #Predefined preset (?)
                tree_preset = st.radio(
                    "Preset:",
                    ["Stable: 50 trees (Ideal untuk klasifikasi penutup lahan dengan kompleksitas rendah)", 
                     "Balanced: 150 trees (Ideal untuk klasifikasi penutup lahan dengan kompleksitas menengah)", 
                     "Complex: 300 trees (Ideal untuk klasifikasi penutup lahan dengan kompleksitas tinggi) "],
                    index=1,
                    help="Semakin banyak jumlah pohon, akurasi dapat meningkat, namun beban komputasi yang semakin tinggi"
                )
                #translate the preset to the machine requirement
                if "Stable" in tree_preset:
                    ntrees = 50                    
                elif "Balanced" in tree_preset:
                    ntrees = 150                   
                else:
                    ntrees = 300
            
            with col2:
                st.markdown("### Pengaturan lainnya")
                st.markdown("*Parameter lainnya menggunakan nilai bawaan (default*")
                
                use_auto_vsplit = True
                v_split = None
                min_leaf = 1
                
                st.success("✅ *Variables per split*: default (akar dari jumlah total variabel)")
                st.success("✅ *Minimum samples*: 1 (default)")
                st.info("💡 Nilai ini umumnya dapat menghasilkan model yang bagus")
        
        with config_tab2:
            st.markdown("**Jika ingin menyesuaikan parameter secara bebas**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🌲 Number of Trees")
                ntrees = st.number_input(
                    "Number of Trees",
                    min_value=10,
                    max_value=900,
                    value=100,
                    step=10,
                    help="Semakin banyak jumlah pohon, akurasi dapat meningkat, namun beban komputasi yang semakin tinggi"
                )
                
            with col2:
                st.markdown("###  Variables per Split")
                use_auto_vsplit = st.checkbox(
                    "Menggunakan nilai bawaan",
                    value=True,
                    help="Menggunakan nilai bawaan berdasarkan data yang digunakan"
                )
                
                if not use_auto_vsplit:
                    v_split = st.number_input(
                        "Variables Per Split",
                        min_value=1,
                        max_value=50,
                        value=5,
                        help="Berapa banyak variabel yang dipertimbangkan saat pengambilan keputusan (split)"
                    )
                else:
                    v_split = None
                    st.success("✅ Menggunakna √(dari jumlah variabel/prediktor)")
            
            with col3:
                st.markdown("### Minimum Leaf Size")
                min_leaf = st.number_input(
                    "Minimum Samples per Leaf",
                    min_value=1,
                    max_value=100,
                    value=1,

                    help= "Jumlah minimal sampel yang dibutuhkan untuk leaf node"
                )
                
        # Ready to train section
        st.markdown("---")
        
        # Show current configuration summary
        with st.expander("📋 Konfigurasi Model", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🌲 Jumlah Pohon", ntrees)
            with col2:
                if v_split is None:
                    st.metric("🔀 Variables per Split", "Default")
                else:
                    st.metric("🔀 Variables per Split", v_split)
            with col3:
                st.metric("🍃 Min Samples per Leaf", min_leaf)
        
        # The big classification button
        if st.button(" Latih Model Klasifikasi", type="primary", use_container_width=True):
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
                
                status_text.text("Saving results...")
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
                
                # Success message with next steps
                st.success("🎉 **Selamat!** Model klasifikasi telah berhasil dilatih!")
                st.info("👉 **Apa selanjutnya?** pergi ke sub-bagian  'Evaluasi Model' untuk melihat performa model klasifikasi!")
                
                # Show a quick preview of what was accomplished
                st.markdown("### ✅ Yang telah dilakukan:")
                st.markdown(f"- ✅ Melatih model Random Forest dengan **{ntrees} pohon keputusan**")
                st.markdown(f"- ✅ Model berusaha untuk mengenali **{len(st.session_state.get('lulc_classes_final', []))} pola penutup lahan yang unik**")
                
            except Exception as e:
                progress_bar.progress(0)
                status_text.text("")
                st.error("❌ **Oops! Something went wrong during training.**")
                st.error("**Error details:** " + str(e))
                
                with st.expander("🔧 Technical Details (for troubleshooting)"):
                    import traceback
                    st.code(traceback.format_exc())
# ==================== TAB 3 Summary Result ====================
with tab3:
    #Lets dump some exposition for this tab
    st.header("Ulasan Model Klasifikasi")
    st.markdown("Platform EPISTEM mendukung dua alat untuk mengulas kemampuan pembelajaran mesin:")
    st.markdown("1. Feature Importance: Kanal mana yang paling penting untuk pembelajaran model?")
    st.markdown("2. Akurasi Model: Bagaimana model menghadapi data yang baru?")
    
    #Column for feature importance and Model accuracy explanation
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Feature Importance**")
        with st.expander("🤔 Apa itu Feature Importance?", expanded=False):
            st.markdown("""
             Analisis tingkat kepentingan fitur merupakan salah satu umpan balik model
             yang bertujuan untuk memberikan informasi kontribusi setiap fitur (dalam konteks ini adalah kanal citra satelit)
             terhadap pembelajaran mesin. Kanal yang memberikan kontribusi paling kecil terhadap model dapat dihilangkan sehingga
             kemampuan pembelajaran model dapat meningkat
            """
                )
    with col2:
        st.markdown("**🎯 Akurasi Model**") 
        with st.expander("🤔 Apa itu evaluasi model?", expanded=False):
            st.markdown("""
            Salah satu kelebihan klasifikasi berbasis pembelajaran mesin adalah kemampuan untuk 
            melakukan evaluasi proses pembelajaran sebelum menghasilkan klasifikasi untuk seluruh citra.
            Evaluasi ini bertujuan untuk melihat bagaimana model melakukan klasifikasi terhadap data yang baru.
            Pendekatan evaluasi ini mirip dengan pengujian akurasi pada peta, namun hal yang membedakan adalah 
            objek yang diuji. Dalam konteks evaluasi model, objek yang diuji adalah prediksi statistik. 
            Jika model belum menghasilkan akurasi yang memuaskan, maka dapat dilakukan pelatihan ulang terhadap model 
            
            """
                )
    
    # Check if classification model is available
    if st.session_state.classification_result is None:
        st.warning("Selesaikan proses pembelajaran model terlebih dahulu!")
        st.stop()
    
    # Check if trained model exists
    if 'trained_model' not in st.session_state:
        st.error("Trained model not found. Please re-run classification.")
        st.stop()
    
    st.divider()
    
    # ==== Feature Importance ====
    st.subheader("📊 Feature Importance Analysis")
    
    with st.expander("Apa yang ditunjukan grafik ini?", expanded=False):
        st.markdown("""
        Grafik ini menunjukan kanal mana yang sangat berguna untuk identifikasi kelas penutup lahan 
        
        - **Nilai yang tinggi** = Lebih penting untuk klasifikasi 
        - **Nilai yang rendah** = Kurang penting untuk proses klasifikasi
        """)
    
    try:
        lulc = Generate_LULC()
        # Get additional parameters for fallback method
        training_data = st.session_state.get('extracted_training_data')
        class_property = st.session_state.get('class_property')
        
        importance_df = lulc.get_feature_importance(
            st.session_state.trained_model,
            training_data=training_data,
            class_property=class_property
        )
        st.session_state.importance_df = importance_df
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Band',
                orientation='h',
                title='Kanal mana yang paling penting?',
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
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Kanal yang paling penting:**")
            for i, row in importance_df.head(5).iterrows():
                st.write(f"{i+1}. {row['Band']}")
            
            with st.expander("View All Bands"):
                st.dataframe(
                    importance_df.style.background_gradient(
                        subset=['Importance'],
                        cmap='YlGn'
                    ),
                    use_container_width=True,
                    hide_index=True
                )
    except Exception as e:
        st.error(f"Could not analyze feature importance: {e}")
    
    st.divider()
    
    # ==== Model Evaluation ====
    st.subheader("🎯 Model Accuracy Assessment")
    
    with st.expander("Bagaimana model diuji?", expanded=False):
        st.markdown("""
        Pengujian model dilakukan dengan menerapkan model kepada data yang tidak digunakan dalam proses pembelajaran
        sehingga kualitas pembelajaran model dapat diketahui. 
        
        **Metric Akurasi:**
        - **Akurasi Keseluruhan/Overall Accuracy**: Persentasi piksels yang diklasifikasikan secara benar
        - **Koefisien Kappa**: Tingkat kesepakatan antara model dan data penguji
        - **F1-Score**: Tingkat rata - rata harmonik antara metrik presisi (precision) dan sensitivitas (sensitivity)
        """)
    #check the model test data avaliability
    have_test_data = st.session_state.extracted_testing_data is not None
    #if its not there
    #user still able to visualize the map
    if not have_test_data:
        st.info("💡 No test data available for accuracy assessment")
        st.markdown("""
        **To evaluate model accuracy:**
        1. Go back to 'Feature Extraction' tab
        2. Check 'Split data into training and testing subsets'
        3. Re-run feature extraction and model training
        4. Return here to see accuracy results
        """)
    #If there's data, capability to calculate model accuracy
    else:
        if st.button("Hitung Akurasi Model", type="primary"):
            with st.spinner("menguji model..."):
                try:
                    lulc = Generate_LULC()
                    class_prop = st.session_state.get('classification_params', {}).get('class_property')
                    
                    model_quality = lulc.evaluate_model(
                        trained_model=st.session_state.trained_model,
                        test_data=st.session_state.extracted_testing_data,
                        class_property=class_prop
                    )
                    
                    st.session_state.model_quality = model_quality
                    st.success("✅ Accuracy assessment complete!")
                    
                except Exception as e:
                    st.error(f"Error during evaluation: {e}")
                    st.code(traceback.format_exc())
        
        # Show results if available
        if "model_quality" in st.session_state:
            st.subheader("📈 Hasil Akurasi Model")
            
            acc = st.session_state.model_quality
            
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                oa = acc['overall_accuracy'] * 100
                st.metric("Overall Accuracy", f"{oa:.1f}%")
            
            with col2:
                kappa = acc['kappa']
                st.metric("Kappa Coefficient", f"{kappa:.3f}")
            
            with col3:
                mean_f1 = sum(acc['f1_scores']) / len(acc['f1_scores'])
                st.metric("Average F1-Score", f"{mean_f1:.3f}")
            
            with col4:
                overall_gmean = acc.get('overall_gmean', 0)
                st.metric("G-Mean Score", f"{overall_gmean:.3f}")
            
            # Interpretation guidelines
            with st.expander("📖 Panduan Interpretasi Hasil", expanded=False):
                st.markdown("""
                **Overall Accuracy (Akurasi Keseluruhan):**
                - **≥ 85%**: Akurasi yang baik untuk sebagian besar aplikasi
                - **70-84%**: Akurasi sedang, mungkin perlu perbaikan
                - **< 70%**: Akurasi rendah, disarankan untuk melatih ulang model
                
                **Kappa Coefficient:**
                - **≥ 0.8**: Kesepakatan yang kuat antara model dan data referensi
                - **0.6-0.79**: Kesepakatan sedang
                - **< 0.6**: Kesepakatan lemah
                
                **F1-Score & G-Mean:**
                - **Nilai mendekati 1.0**: Performa yang baik
                - **Nilai mendekati 0.5**: Performa sedang
                - **Nilai mendekati 0.0**: Performa rendah
                
                💡 **Catatan:** Interpretasi ini bersifat umum. Standar akurasi dapat bervariasi tergantung pada aplikasi dan kompleksitas area studi.
                """)
            
            st.markdown("---")
            
            # Class-level results
            st.subheader("📋 Results by Land Cover Class")
            
            df_metrics = pd.DataFrame({
                "Class ID": range(len(acc["precision"])),
                "Producer's Accuracy (%)": np.round(np.array(acc["recall"]) * 100, 1),
                "User's Accuracy (%)": np.round(np.array(acc["precision"]) * 100, 1),
                "F1-Score (%)": np.round(np.array(acc["f1_scores"]) * 100, 1),
                "G-Mean Score (%)": np.round(np.array(acc["gmean_per_class"]) * 100, 1)
            })
            
            st.dataframe(df_metrics, use_container_width=True)
            
            # Confusion Matrix
            st.subheader("🔍 Confusion Matrix")
            st.markdown("Shows how often each class was correctly identified vs confused with other classes")
            
            cm = pd.DataFrame(
                acc["confusion_matrix"],
                columns=[f"Predicted {i}" for i in range(len(acc["confusion_matrix"]))],
                index=[f"Actual {i}" for i in range(len(acc["confusion_matrix"]))]
            )
            
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                title="Confusion Matrix: Actual vs Predicted Classes"
            )
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)

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
                
                # Validate that classification_map is an ee.Image
                if not isinstance(classification_map, ee.Image):
                    st.error(f"Invalid classification result type: {type(classification_map)}. Expected ee.Image.")
                    st.error(f"Classification result content: {classification_map}")
                    st.stop()
                
                # Additional validation - try to get basic info about the image
                try:
                    # Test if we can get band names (this will fail if it's not a proper ee.Image)
                    band_names = classification_map.bandNames().getInfo()
                    st.info(f"Classification image bands: {band_names}")
                except Exception as validation_error:
                    st.error(f"Classification result validation failed: {validation_error}")
                    st.error("The classification result appears to be corrupted or invalid.")
                    st.stop()
                
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
                
                # Add layers with error handling
                try:
                    Map.addLayer(classification_map, vis_params, 'Random Forest Classification', True)
                    st.success("✅ Classification layer added successfully")
                except Exception as layer_error:
                    st.error(f"Failed to add classification layer: {layer_error}")
                    st.error("This usually indicates the classification result contains invalid data.")
                    
                    # Try to get more info about the classification_map
                    try:
                        # Check if it's actually an ee.Image and get some basic info
                        if hasattr(classification_map, 'getInfo'):
                            info = classification_map.getInfo()
                            st.error(f"Classification map info: {info}")
                        else:
                            st.error(f"Classification map is not an Earth Engine object: {type(classification_map)}")
                    except Exception as info_error:
                        st.error(f"Could not get classification map info: {info_error}")
                    
                    st.stop()
                
                try:
                    Map.addLayer(image, st.session_state.get('visualization', {}), 'Image Composite', False)
                except Exception as image_error:
                    st.warning(f"Could not add image composite layer: {image_error}")
                    # Continue anyway since the main issue is with classification_map
                
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
    if st.button("⬅️ Back to Module 3: Analyze ROI", use_container_width=True):
        st.switch_page("pages/4_Module_4_Analyze_ROI.py")

with col2:
    if st.session_state.classification_result is not None:
        if st.button("➡️ Go to Module 7: Thematic Accuracy Assessment", use_container_width=True):
            st.switch_page("pages/6_Module_7_Thematic_Accuracy.py")
            st.info("Accuracy assessment module coming soon!")
    else:
        st.button("🔒 Complete Classification First", disabled=True, use_container_width=True)

# Show completion status
if st.session_state.classification_result is not None:
    st.success(f"Classification completed using {st.session_state.get('classification_mode', 'N/A')}")
else:
    st.info("💡 Complete feature extraction and classification to proceed")