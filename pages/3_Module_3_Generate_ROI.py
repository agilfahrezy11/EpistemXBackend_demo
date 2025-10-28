import streamlit as st
import pandas as pd
import geopandas as gpd
import ee
from epistemx.module_3 import InputCheck, SyncTrainData, SplitTrainData, LULCSamplingTool
from epistemx.ee_config import initialize_earth_engine

# Initialize Earth Engine
initialize_earth_engine()

# Module title and description
st.title("Penentuan Data Sampel Klasifikasi Tutupan/penggunaan lahan")
st.divider()
st.markdown("Modul ini memungkinkan Anda untuk menyiapkan dan menentukan data sampel yang digunakan untuk proses klasifikasi tutupan/penggunaan lahan. "\
    "Untuk menggunakan modul ini, hasil dari modul 1 dan 2 harus sudah tersedia. Jika sudah terpenuhi, Anda dapat:"\
    )
st.markdown("1. Mengunggah data sampel training.")
st.markdown("2. Membuat data sampel training melalui sampling on screen.")
st.markdown("3. Menggunakan data sampel default Epistem.")

# Module description
markdown = """
Modul ini dibuat untuk menentukan data sampel training.
"""

# Set page layout and side info
st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "logos\logo_epistem.png"
st.sidebar.image(logo)

st.markdown("Ketersediaan keluaran hasil modul 1 dan 2")

# Check prerequisites
col1, col2 = st.columns(2)

with col1:
    # Check AOI availability
    if 'AOI' in st.session_state and 'gdf' in st.session_state:
        st.success("✅ Data AOI dari modul 1 tersedia")
        aoi_available = True
    else:
        st.error("❌ Data AOI belum tersedia, silakan kunjungi modul 1.")
        aoi_available = False

with col2:
    # Check classification scheme - check for both possible data sources
    classification_available = (
        ('classification_df' in st.session_state and not st.session_state['classification_df'].empty) or
        ('lulc_classes_final' in st.session_state and len(st.session_state['lulc_classes_final']) > 0) or
        ('classes' in st.session_state and len(st.session_state['classes']) > 0)
    )
    
    if classification_available:
        # Determine scheme type and class count
        if 'classification_df' in st.session_state and not st.session_state['classification_df'].empty:
            class_count = len(st.session_state['classification_df'])
        elif 'lulc_classes_final' in st.session_state:
            class_count = len(st.session_state['lulc_classes_final'])
        else:
            class_count = len(st.session_state.get('classes', []))
            
        scheme_type = "Default Scheme" if st.session_state.get('ReferenceDataSource', False) else "Custom Scheme"
        st.success(f"✅ Data skema klasifikasi dari modul 2 tersedia ({scheme_type}) - {class_count} kelas")
        scheme_available = True
    else:
        st.error("❌ Data skema klasifikasi belum tersedia, silakan kunjungi modul 2.")
        scheme_available = False

if not (aoi_available and scheme_available):
    st.stop()

# Get data from previous modules
AOI = st.session_state.get('AOI')  # From Module 1 (EE object)
AOI_GDF = st.session_state.get('gdf')  # From Module 1 (GeoDataFrame)

# Get classification data from Module 2 - try multiple sources for compatibility
if 'classification_df' in st.session_state and not st.session_state['classification_df'].empty:
    # Use the formatted DataFrame from Module 2
    LULCTable = st.session_state['classification_df'].copy()
    # Ensure consistent column names for Module 3
    if 'Land Cover Class' in LULCTable.columns:
        LULCTable = LULCTable.rename(columns={'Land Cover Class': 'LULC_Type'})
elif 'lulc_classes_final' in st.session_state and len(st.session_state['lulc_classes_final']) > 0:
    # Convert from manager classes format
    classes_data = []
    for cls in st.session_state['lulc_classes_final']:
        classes_data.append({
            'ID': cls.get('ID', cls.get('Class ID', '')),
            'LULC_Type': cls.get('Class Name', cls.get('Land Cover Class', '')),
            'Color Palette': cls.get('Color Code', cls.get('Color Palette', cls.get('Color', '#2e8540')))
        })
    LULCTable = pd.DataFrame(classes_data)
elif 'classes' in st.session_state and len(st.session_state['classes']) > 0:
    # Fallback to legacy format
    LULCTable = pd.DataFrame(st.session_state['classes'])
else:
    # No classification data available
    LULCTable = pd.DataFrame()

# Additional validation for LULCTable
if LULCTable.empty:
    st.error("❌ Data klasifikasi tidak dapat dimuat dengan benar. Silakan kembali ke Modul 2 dan pastikan skema klasifikasi telah disimpan.")
    st.stop()

# Check ReferenceDataSource from Module 2
reference_data_source = st.session_state.get('ReferenceDataSource', False)

# Debug information - uncomment to troubleshoot
# with st.expander("🔧 Debug Information"):
#     st.write(f"ReferenceDataSource: {st.session_state.get('ReferenceDataSource', 'Not set')}")
#     st.write(f"Classification DF available: {'classification_df' in st.session_state}")
#     st.write(f"LULC Classes Final available: {'lulc_classes_final' in st.session_state}")
#     st.write(f"Legacy classes available: {'classes' in st.session_state}")
#     if not LULCTable.empty:
#         st.write(f"LULCTable shape: {LULCTable.shape}")
#         st.write(f"LULCTable columns: {list(LULCTable.columns)}")
#         st.dataframe(LULCTable.head())
#     else:
#         st.write("LULCTable is empty!")
#     st.write(f"AOI type: {type(AOI)}")
#     st.write(f"AOI_GDF type: {type(AOI_GDF)}")
#     if AOI_GDF is not None:
#         st.write(f"AOI_GDF CRS: {AOI_GDF.crs}")
#         st.write(f"AOI_GDF shape: {AOI_GDF.shape}")

st.divider()

# Initialize variables
TrainField = 'LULC_Type'  # Default field name
UploadTrainData = False
TrainDataDict = None
TrainDataRecap = pd.DataFrame()

if reference_data_source:
    # User chose default scheme in Module 2 - automatically use default training data
    st.info("🔄 Berdasarkan pilihan skema klasifikasi default di Modul 2, sistem akan menggunakan data sampel default Epistem.")
    st.subheader("Gunakan data sampel default (Epistem)")
    st.markdown("Data training akan dimuat dari dataset referensi RESTORE+ yang sesuai dengan skema klasifikasi yang dipilih.")
    
    TrainEePath = 'projects/ee-rg2icraf/assets/Indonesia_lulc_Sample'
    class_col_index = 0
    TrainField = 'kelas'
    UploadTrainData = False
    
    # Load reference training data
    if st.button("Muat Data Training Referensi", type="primary"):
        try:
            with st.spinner("Memuat data training referensi..."):
                # Initialize TrainDataDict
                TrainDataDict = {
                    'training_data': None,
                    'landcover_df': LULCTable,
                    'class_field': TrainField,
                    'validation_results': {
                        'total_points': 0,
                        'valid_points': 0,
                        'points_after_class_filter': 0,
                        'invalid_classes': [],
                        'outside_aoi': [],
                        'insufficient_samples': [],
                        'warnings': []
                    }
                }
                
                # Load training data
                TrainDataDict = SyncTrainData.LoadTrainData(
                    landcover_df=LULCTable,
                    aoi_geometry=AOI,
                    training_shp_path=None,
                    training_ee_path=TrainEePath
                )
                
                # Update the aoi_geometry in TrainDataDict to use GeoDataFrame for filtering
                if AOI_GDF is not None:
                    TrainDataDict['aoi_geometry'] = AOI_GDF
                
                st.session_state['train_data_dict_ref'] = TrainDataDict
                st.session_state['reference_data_loaded'] = True
                st.success("Data training referensi berhasil dimuat!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error memuat data training referensi: {e}")
    
    # Process reference data if loaded
    if st.session_state.get('reference_data_loaded', False):
        st.divider()
        st.subheader("B. Pemrosesan & Validasi Data Training")
        
        TrainDataDict = st.session_state.get('train_data_dict_ref')
        
        if st.button("Proses Data Training", type="primary", key="process_reference_data"):
            try:
                progress = st.progress(0)
                status_text = st.empty()
                
                # Set class field
                status_text.text("Langkah 1/5: Mengatur field kelas...")
                if TrainDataDict.get('training_data') is not None:
                    TrainDataDict = SyncTrainData.SetClassField(TrainDataDict, TrainField)
                progress.progress(20)
                
                # Validate classes
                status_text.text("Langkah 2/5: Memvalidasi kelas...")
                if TrainDataDict.get('training_data') is not None:
                    TrainDataDict = SyncTrainData.ValidClass(TrainDataDict, use_class_ids=True)
                progress.progress(40)
                
                # Check sufficiency
                status_text.text("Langkah 3/5: Memeriksa kecukupan sampel...")
                if TrainDataDict.get('training_data') is not None:
                    TrainDataDict = SyncTrainData.CheckSufficiency(TrainDataDict, min_samples=20)
                progress.progress(60)
                
                # Filter by AOI
                status_text.text("Langkah 4/5: Memfilter berdasarkan AOI...")
                if TrainDataDict.get('training_data') is not None:
                    if AOI_GDF is not None:
                        TrainDataDict['aoi_geometry'] = AOI_GDF
                    TrainDataDict = SyncTrainData.FilterTrainAoi(TrainDataDict)
                progress.progress(80)
                
                # Create summary table
                status_text.text("Langkah 5/5: Membuat ringkasan...")
                if TrainDataDict.get('training_data') is not None:
                    table_df, total_samples, insufficient_df = SyncTrainData.TrainDataRaw(
                        training_data=TrainDataDict.get('training_data'),
                        landcover_df=TrainDataDict.get('landcover_df'),
                        class_field=TrainDataDict.get('class_field')
                    )
                    
                    st.session_state['table_df_ref'] = table_df
                    st.session_state['total_samples_ref'] = total_samples
                    st.session_state['insufficient_df_ref'] = insufficient_df
                    st.session_state['train_data_final_ref'] = TrainDataDict.get('training_data')
                    st.session_state['train_data_dict_ref'] = TrainDataDict
                
                progress.progress(100)
                status_text.text("Pemrosesan selesai!")
                st.session_state['data_processed_ref'] = True
                st.success("Data training berhasil diproses!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error memproses data training: {e}")
                import traceback
                st.error("Full traceback:")
                st.code(traceback.format_exc())
        
        # Display Results for Reference Data
        if st.session_state.get('data_processed_ref', False):
            st.divider()
            st.subheader("C. Ringkasan Data Training")
            
            # Get validation results
            TrainDataDict = st.session_state.get('train_data_dict_ref', {})
            vr = TrainDataDict.get('validation_results', {})
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Titik Dimuat", vr.get('total_points', 'N/A'))
            with col2:
                st.metric("Titik Setelah Filter Kelas", vr.get('points_after_class_filter', 'N/A'))
            with col3:
                st.metric("Titik Valid (dalam AOI)", vr.get('valid_points', 'N/A'))
            with col4:
                st.metric("Kelas Invalid", len(vr.get('invalid_classes', [])))
            
            # Training data distribution table
            if 'table_df_ref' in st.session_state and st.session_state['table_df_ref'] is not None:
                st.markdown("#### Distribusi Data Training")
                
                # Format percentage column
                display_df = st.session_state['table_df_ref'].copy()
                if 'Percentage' in display_df.columns:
                    display_df['Percentage'] = display_df['Percentage'].apply(
                        lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
                    )
                
                st.dataframe(display_df,width='stretch')
                
                # Show insufficient classes if any
                if 'insufficient_df_ref' in st.session_state and st.session_state['insufficient_df_ref'] is not None:
                    if len(st.session_state['insufficient_df_ref']) > 0:
                        st.warning(f"⚠️ {len(st.session_state['insufficient_df_ref'])} kelas memiliki sampel yang tidak mencukupi (< 20 sampel)")
                        
                        with st.expander("Lihat Kelas yang Tidak Mencukupi"):
                            insufficient_display = st.session_state['insufficient_df_ref'].copy()
                            if 'Percentage' in insufficient_display.columns:
                                insufficient_display['Percentage'] = insufficient_display['Percentage'].apply(
                                    lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
                                )
                            st.dataframe(insufficient_display,width='stretch')
            
            # Training/Validation Split for Reference Data
            st.divider()
            st.subheader("D. Pembagian Data Training/Validasi")
            
            split_data = st.checkbox("Bagi data menjadi set training dan validasi", value=True, key="split_ref")
            
            if split_data:
                split_ratio = st.slider("Persentase data training:", min_value=0.5, max_value=0.9, value=0.7, step=0.05, key="split_ratio_ref")
                
                if st.button("Bagi Data Training", type="primary", key="split_ref_data"):
                    try:
                        if 'train_data_final_ref' in st.session_state and st.session_state['train_data_final_ref'] is not None:
                            train_data = st.session_state['train_data_final_ref']
                            
                            # Perform split
                            train_final, valid_final = SplitTrainData.SplitProcess(
                                train_data,
                                TrainSplitPct=split_ratio,
                                random_state=123
                            )
                            
                            st.session_state['train_final_ref'] = train_final
                            st.session_state['valid_final_ref'] = valid_final
                            st.session_state['split_completed_ref'] = True
                            
                            # Display split statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Sampel Training", len(train_final))
                            with col2:
                                st.metric("Sampel Validasi", len(valid_final))
                            with col3:
                                st.metric("Total Sampel", len(train_data))
                            
                            st.success("Pembagian data berhasil diselesaikan!")
                            
                            # Show preview of split data
                            with st.expander("Preview Data yang Dibagi"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Preview Data Training:**")
                                    st.dataframe(train_final.head())
                                with col2:
                                    st.markdown("**Preview Data Validasi:**")
                                    st.dataframe(valid_final.head())
                                    
                    except Exception as e:
                        st.error(f"Error membagi data: {e}")
            else:
                # If not splitting, use all data as training
                if 'train_data_final_ref' in st.session_state and st.session_state['train_data_final_ref'] is not None:
                    st.session_state['train_final_ref'] = st.session_state['train_data_final_ref']
                    st.session_state['valid_final_ref'] = None
                    st.session_state['split_completed_ref'] = True
                    st.info("Menggunakan seluruh dataset sebagai data training.")

else:
    # User chose manual/CSV scheme - show both options directly
    st.subheader("Pilih Metode Pengumpulan Data Training")
    
    # Create tabs for the two options
    tab1, tab2 = st.tabs(["📤 Unggah Data Sampel", "🎯 Sampling On Screen"])
    
    class_col_index = 1

    with tab1:
        # Option 1: Upload sample data
        st.subheader("A. Unggah data sampel (Shapefile)")
        st.markdown("Silakan unggah data shapefile terkompresi dalam format .zip")
        uploaded_file = st.file_uploader("Unggah shapefile (.zip)", type=["zip"])
        
        if uploaded_file:
            # Handle shapefile upload (implementation from Training_Data_Collection.py)
            import tempfile
            import zipfile
            import os
            
            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract zip file
                zip_path = os.path.join(tmpdir, "training.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(tmpdir)
                
                # Find shapefile
                shp_files = []
                for root, _, files in os.walk(tmpdir):
                    for fname in files:
                        if fname.lower().endswith(".shp"):
                            shp_files.append(os.path.join(root, fname))
                
                if shp_files:
                    try:
                        # Load and validate shapefile
                        import geopandas as gpd
                        gdf = gpd.read_file(shp_files[0])
                        st.success("Data training berhasil dimuat!")
                        
                        # Show preview
                        st.markdown("**Preview data training:**")
                        st.dataframe(gdf.head())
                        
                        # Field selection
                        TrainField = st.selectbox(
                            "Pilih field yang berisi informasi kelas:",
                            options=gdf.columns.tolist(),
                            help="Field ini harus berisi ID kelas atau nama yang sesuai dengan skema klasifikasi Anda"
                        )
                        
                        if st.button("Proses Data Training", type="primary", key="process_uploaded_data"):
                            st.session_state['training_gdf'] = gdf
                            st.session_state['training_class_field'] = TrainField
                            st.session_state['upload_data_loaded'] = True
                            UploadTrainData = True
                            st.success("Data training berhasil diproses!")
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"Error membaca shapefile: {e}")
                else:
                    st.error("File .shp tidak ditemukan dalam zip yang diunggah.")
        
        # Process uploaded data if loaded
        if st.session_state.get('upload_data_loaded', False):
            st.divider()
            st.subheader("B. Pemrosesan & Validasi Data Training")
            
            gdf = st.session_state.get('training_gdf')
            TrainField = st.session_state.get('training_class_field')
            
            TrainDataDict = {
                'training_data': gdf,
                'landcover_df': LULCTable,
                'class_field': TrainField,
                'validation_results': {
                    'total_points': len(gdf),
                    'valid_points': 0,
                    'points_after_class_filter': 0,
                    'invalid_classes': [],
                    'outside_aoi': [],
                    'insufficient_samples': [],
                    'warnings': []
                }
            }
            
            if st.button("Proses Data Training", type="primary", key="process_upload_validation"):
                try:
                    progress = st.progress(0)
                    status_text = st.empty()
                    
                    # Set class field
                    status_text.text("Langkah 1/5: Mengatur field kelas...")
                    if TrainDataDict.get('training_data') is not None:
                        TrainDataDict = SyncTrainData.SetClassField(TrainDataDict, TrainField)
                    progress.progress(20)
                    
                    # Validate classes
                    status_text.text("Langkah 2/5: Memvalidasi kelas...")
                    if TrainDataDict.get('training_data') is not None:
                        TrainDataDict = SyncTrainData.ValidClass(TrainDataDict, use_class_ids=False)
                    progress.progress(40)
                    
                    # Check sufficiency
                    status_text.text("Langkah 3/5: Memeriksa kecukupan sampel...")
                    if TrainDataDict.get('training_data') is not None:
                        TrainDataDict = SyncTrainData.CheckSufficiency(TrainDataDict, min_samples=20)
                    progress.progress(60)
                    
                    # Filter by AOI
                    status_text.text("Langkah 4/5: Memfilter berdasarkan AOI...")
                    if TrainDataDict.get('training_data') is not None:
                        if AOI_GDF is not None:
                            TrainDataDict['aoi_geometry'] = AOI_GDF
                        TrainDataDict = SyncTrainData.FilterTrainAoi(TrainDataDict)
                    progress.progress(80)
                    
                    # Create summary table
                    status_text.text("Langkah 5/5: Membuat ringkasan...")
                    if TrainDataDict.get('training_data') is not None:
                        table_df, total_samples, insufficient_df = SyncTrainData.TrainDataRaw(
                            training_data=TrainDataDict.get('training_data'),
                            landcover_df=TrainDataDict.get('landcover_df'),
                            class_field=TrainDataDict.get('class_field')
                        )
                        
                        st.session_state['table_df_upload'] = table_df
                        st.session_state['total_samples_upload'] = total_samples
                        st.session_state['insufficient_df_upload'] = insufficient_df
                        st.session_state['train_data_final_upload'] = TrainDataDict.get('training_data')
                        st.session_state['train_data_dict_upload'] = TrainDataDict
                    
                    progress.progress(100)
                    status_text.text("Pemrosesan selesai!")
                    st.session_state['data_processed_upload'] = True
                    st.success("Data training berhasil diproses!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error memproses data training: {e}")
                    import traceback
                    st.error("Full traceback:")
                    st.code(traceback.format_exc())
            
            # Display Results for Uploaded Data
            if st.session_state.get('data_processed_upload', False):
                st.divider()
                st.subheader("C. Ringkasan Data Training")
                
                # Get validation results
                TrainDataDict = st.session_state.get('train_data_dict_upload', {})
                vr = TrainDataDict.get('validation_results', {})
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Titik Dimuat", vr.get('total_points', 'N/A'))
                with col2:
                    st.metric("Titik Setelah Filter Kelas", vr.get('points_after_class_filter', 'N/A'))
                with col3:
                    st.metric("Titik Valid (dalam AOI)", vr.get('valid_points', 'N/A'))
                with col4:
                    st.metric("Kelas Invalid", len(vr.get('invalid_classes', [])))
                
                # Training data distribution table
                if 'table_df_upload' in st.session_state and st.session_state['table_df_upload'] is not None:
                    st.markdown("#### Distribusi Data Training")
                    
                    # Format percentage column
                    display_df = st.session_state['table_df_upload'].copy()
                    if 'Percentage' in display_df.columns:
                        display_df['Percentage'] = display_df['Percentage'].apply(
                            lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
                        )
                    
                    st.dataframe(display_df,width='stretch')
                    
                    # Show insufficient classes if any
                    if 'insufficient_df_upload' in st.session_state and st.session_state['insufficient_df_upload'] is not None:
                        if len(st.session_state['insufficient_df_upload']) > 0:
                            st.warning(f"⚠️ {len(st.session_state['insufficient_df_upload'])} kelas memiliki sampel yang tidak mencukupi (< 20 sampel)")
                            
                            with st.expander("Lihat Kelas yang Tidak Mencukupi"):
                                insufficient_display = st.session_state['insufficient_df_upload'].copy()
                                if 'Percentage' in insufficient_display.columns:
                                    insufficient_display['Percentage'] = insufficient_display['Percentage'].apply(
                                        lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
                                    )
                                st.dataframe(insufficient_display,width='stretch')
                
                # Training/Validation Split for Uploaded Data
                st.divider()
                st.subheader("D. Pembagian Data Training/Validasi")
                
                split_data = st.checkbox("Bagi data menjadi set training dan validasi", value=True, key="split_upload")
                
                if split_data:
                    split_ratio = st.slider("Persentase data training:", min_value=0.5, max_value=0.9, value=0.7, step=0.05, key="split_ratio_upload")
                    
                    if st.button("Bagi Data Training", type="primary", key="split_upload_data"):
                        try:
                            if 'train_data_final_upload' in st.session_state and st.session_state['train_data_final_upload'] is not None:
                                train_data = st.session_state['train_data_final_upload']
                                
                                # Perform split
                                train_final, valid_final = SplitTrainData.SplitProcess(
                                    train_data,
                                    TrainSplitPct=split_ratio,
                                    random_state=123
                                )
                                
                                st.session_state['train_final_upload'] = train_final
                                st.session_state['valid_final_upload'] = valid_final
                                st.session_state['split_completed_upload'] = True
                                
                                # Display split statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Sampel Training", len(train_final))
                                with col2:
                                    st.metric("Sampel Validasi", len(valid_final))
                                with col3:
                                    st.metric("Total Sampel", len(train_data))
                                
                                st.success("Pembagian data berhasil diselesaikan!")
                                
                                # Show preview of split data
                                with st.expander("Preview Data yang Dibagi"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Preview Data Training:**")
                                        st.dataframe(train_final.head())
                                    with col2:
                                        st.markdown("**Preview Data Validasi:**")
                                        st.dataframe(valid_final.head())
                                        
                        except Exception as e:
                            st.error(f"Error membagi data: {e}")
                else:
                    # If not splitting, use all data as training
                    if 'train_data_final_upload' in st.session_state and st.session_state['train_data_final_upload'] is not None:
                        st.session_state['train_final_upload'] = st.session_state['train_data_final_upload']
                        st.session_state['valid_final_upload'] = None
                        st.session_state['split_completed_upload'] = True
                        st.info("Menggunakan seluruh dataset sebagai data training.")

    with tab2:
        # Option 2: Create sample data on screen
        st.subheader("A. Buat data sampel (On Screen)")
        st.info("Gunakan peta di bawah untuk mengumpulkan sampel training dengan menambahkan koordinat secara manual.")
        
        # Initialize sampling data in session state
        if 'sampling_data' not in st.session_state:
            st.session_state['sampling_data'] = []
        if 'current_sampling_class' not in st.session_state:
            st.session_state['current_sampling_class'] = None
        
        # Class selection for sampling - use the same LULCTable as the rest of the module
        classes_df = LULCTable.copy()
        
        if not classes_df.empty:
            # Ensure consistent column names
            if 'Land Cover Class' in classes_df.columns:
                classes_df = classes_df.rename(columns={'Land Cover Class': 'LULC_Type'})
            if 'Color Palette' in classes_df.columns:
                classes_df = classes_df.rename(columns={'Color Palette': 'color_palette'})
            
            # Use column positions - ensure we have the required columns
            id_col = 'ID' if 'ID' in classes_df.columns else classes_df.columns[0]
            type_col = 'LULC_Type' if 'LULC_Type' in classes_df.columns else classes_df.columns[1] if len(classes_df.columns) > 1 else classes_df.columns[0]
            color_col = 'color_palette' if 'color_palette' in classes_df.columns else (classes_df.columns[2] if len(classes_df.columns) > 2 else None)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_class_idx = st.selectbox(
                    "Pilih kelas untuk sampling:",
                    options=range(len(classes_df)),
                    format_func=lambda x: f"{classes_df.iloc[x][id_col]}: {classes_df.iloc[x][type_col]}",
                    key="sampling_class_selector"
                )
                
                if selected_class_idx is not None:
                    selected_class = classes_df.iloc[selected_class_idx]
                    st.session_state['current_sampling_class'] = {
                        'ID': selected_class[id_col],
                        'LULC_Type': selected_class[type_col],
                        'color_palette': selected_class[color_col] if color_col and color_col in selected_class else '#FF0000'
                    }
                    st.info(f"Kelas aktif: {selected_class[id_col]} - {selected_class[type_col]}")
            
            with col2:
                if st.button("Hapus Semua Titik", type="secondary"):
                    st.session_state['sampling_data'] = []
                    st.success("Semua titik sampling telah dihapus!")
                    st.rerun()
            
            # Display current sampling statistics
            if st.session_state['sampling_data']:
                sampling_df = pd.DataFrame(st.session_state['sampling_data'])
                stats_summary = sampling_df.groupby(['class_id', 'class_type']).size().reset_index(name='count')
                
                st.markdown("**Statistik Sampling Saat Ini:**")
                for _, row in stats_summary.iterrows():
                    st.write(f"- {row['class_type']}: {row['count']} titik")
            
            # Interactive map for sampling
            try:
                if AOI_GDF is not None and not AOI_GDF.empty:
                    # Create map centered on AOI
                    centroid = AOI_GDF.geometry.centroid.iloc[0]
                    center_lat, center_lon = centroid.y, centroid.x
                    
                    # Create geemap map with satellite basemap
                    import geemap.foliumap as geemap
                    m = geemap.Map(center=[center_lat, center_lon], zoom=10)
                    
                    # Change basemap to satellite imagery (free)
                    m.add_basemap('SATELLITE')
                    
                    # Add AOI boundary
                    m.add_geojson(AOI_GDF.__geo_interface__, layer_name="AOI Boundary", style={'color': 'blue', 'fillOpacity': 0.1})
                    
                    # Add existing sampling points if any
                    if st.session_state['sampling_data']:
                        for point in st.session_state['sampling_data']:
                            m.add_marker(
                                location=[point['latitude'], point['longitude']], 
                                popup=f"Class: {point['class_type']}"
                            )
                    
                    # Display map
                    m.to_streamlit(height=600, add_layer_control=True)
                    
                    # Manual coordinate input
                    st.markdown("**Tambahkan titik secara manual:**")
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        manual_lat = st.number_input("Latitude:", value=center_lat, format="%.6f", key="manual_lat")
                    with col2:
                        manual_lon = st.number_input("Longitude:", value=center_lon, format="%.6f", key="manual_lon")
                    with col3:
                        if st.button("Tambah Titik Manual", type="primary"):
                            if st.session_state['current_sampling_class'] is not None:
                                # Check if point is within AOI
                                from shapely.geometry import Point
                                point_geom = Point(manual_lon, manual_lat)
                                
                                if AOI_GDF.geometry.contains(point_geom).any():
                                    new_point = {
                                        'latitude': manual_lat,
                                        'longitude': manual_lon,
                                        'class_id': st.session_state['current_sampling_class']['ID'],
                                        'class_type': st.session_state['current_sampling_class']['LULC_Type'],
                                        'color': st.session_state['current_sampling_class']['color_palette']
                                    }
                                    st.session_state['sampling_data'].append(new_point)
                                    st.success(f"Titik berhasil ditambahkan pada ({manual_lat:.6f}, {manual_lon:.6f})")
                                    st.rerun()
                                else:
                                    st.error("Titik berada di luar batas AOI!")
                            else:
                                st.error("Silakan pilih kelas terlebih dahulu!")
                    
                    # # Bulk coordinate input
                    # st.markdown("**Atau tambahkan beberapa titik sekaligus:**")
                    # coord_text = st.text_area(
                    #     "Masukkan koordinat (satu per baris, format: latitude,longitude):",
                    #     placeholder="Contoh:\n-2.123456,106.789012\n-2.234567,106.890123",
                    #     height=100
                    # )
                    
                    # if st.button("Tambah Titik Bulk", type="primary"):
                    #     if st.session_state['current_sampling_class'] is not None and coord_text.strip():
                    #         lines = coord_text.strip().split('\n')
                    #         added_count = 0
                            
                    #         for line in lines:
                    #             line = line.strip()
                    #             if line and ',' in line:
                    #                 try:
                    #                     lat_str, lon_str = line.split(',')
                    #                     lat = float(lat_str.strip())
                    #                     lon = float(lon_str.strip())
                                        
                    #                     # Check if point is within AOI
                    #                     from shapely.geometry import Point
                    #                     point_geom = Point(lon, lat)
                                        
                    #                     if AOI_GDF.geometry.contains(point_geom).any():
                    #                         new_point = {
                    #                             'latitude': lat,
                    #                             'longitude': lon,
                    #                             'class_id': st.session_state['current_sampling_class']['ID'],
                    #                             'class_type': st.session_state['current_sampling_class']['LULC_Type'],
                    #                             'color': st.session_state['current_sampling_class']['color_palette']
                    #                         }
                    #                         st.session_state['sampling_data'].append(new_point)
                    #                         added_count += 1
                    #                     else:
                    #                         st.warning(f"Titik ({lat:.6f}, {lon:.6f}) berada di luar AOI dan dilewati")
                    #                 except ValueError:
                    #                     st.warning(f"Format koordinat tidak valid: {line}")
                            
                    #         if added_count > 0:
                    #             st.success(f"Berhasil menambahkan {added_count} titik!")
                    #             st.rerun()
                    #     else:
                    #         st.error("Silakan pilih kelas dan masukkan koordinat!")
                    
                    # Process collected samples
                    if st.session_state['sampling_data']:
                        st.divider()
                        st.subheader("B. Pemrosesan & Validasi Data Training")
                        
                        if st.button("Proses Sampel yang Dikumpulkan", type="primary", key="process_sampling_data"):
                            try:
                                # Convert to GeoDataFrame
                                from shapely.geometry import Point
                                
                                geometries = [Point(point['longitude'], point['latitude']) for point in st.session_state['sampling_data']]
                                
                                # Create training data structure
                                training_points = []
                                for point in st.session_state['sampling_data']:
                                    training_points.append({
                                        'kelas': point['class_id'],
                                        'LULC_Type': point['class_type'],
                                        'latitude': point['latitude'],
                                        'longitude': point['longitude']
                                    })
                                
                                train_data_gdf = gpd.GeoDataFrame(training_points, geometry=geometries, crs='EPSG:4326')
                                
                                # Store in session state
                                st.session_state['train_data_final_sampling'] = train_data_gdf
                                st.session_state['data_processed_sampling'] = True
                                
                                # Create summary table
                                summary_data = []
                                for _, class_row in classes_df.iterrows():
                                    class_points = [p for p in st.session_state['sampling_data'] if p['class_id'] == class_row[id_col]]
                                    
                                    summary_data.append({
                                        'ID': class_row[id_col],
                                        'LULC_class': class_row[type_col],
                                        'Sample_Count': len(class_points),
                                        'Percentage': (len(class_points) / len(st.session_state['sampling_data']) * 100) if st.session_state['sampling_data'] else 0,
                                        'Status': 'Sufficient' if len(class_points) >= 20 else 'Insufficient' if len(class_points) > 0 else 'No Samples'
                                    })
                                
                                st.session_state['table_df_sampling'] = pd.DataFrame(summary_data)
                                st.session_state['total_samples_sampling'] = len(st.session_state['sampling_data'])
                                
                                st.success(f"Berhasil memproses {len(train_data_gdf)} titik training dari sampling!")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error memproses sampel: {e}")
                                st.error(f"Detail error: {str(e)}")
                        
                        # Display Results for Sampling Data
                        if st.session_state.get('data_processed_sampling', False):
                            st.divider()
                            st.subheader("C. Ringkasan Data Training")
                            
                            # Summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            total_samples = st.session_state.get('total_samples_sampling', 0)
                            unique_classes = len(set([p['class_id'] for p in st.session_state['sampling_data']]))
                            avg_samples = total_samples / unique_classes if unique_classes > 0 else 0
                            insufficient_classes = len([row for _, row in st.session_state.get('table_df_sampling', pd.DataFrame()).iterrows() if row['Status'] in ['Insufficient', 'No Samples']])
                            
                            with col1:
                                st.metric("Total Sampel", total_samples)
                            with col2:
                                st.metric("Kelas Unik", unique_classes)
                            with col3:
                                st.metric("Rata-rata Sampel/Kelas", f"{avg_samples:.1f}")
                            with col4:
                                st.metric("Kelas Tidak Cukup", insufficient_classes)
                            
                            # Training data distribution table
                            if 'table_df_sampling' in st.session_state and st.session_state['table_df_sampling'] is not None:
                                st.markdown("#### Distribusi Data Training")
                                
                                # Format percentage column
                                display_df = st.session_state['table_df_sampling'].copy()
                                if 'Percentage' in display_df.columns:
                                    display_df['Percentage'] = display_df['Percentage'].apply(
                                        lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
                                    )
                                
                                st.dataframe(display_df,width='stretch')
                                
                                # Show insufficient classes if any
                                insufficient_df = display_df[display_df['Status'].isin(['Insufficient', 'No Samples'])]
                                if not insufficient_df.empty:
                                    st.warning(f"⚠️ {len(insufficient_df)} kelas memiliki sampel yang tidak mencukupi (< 20 sampel)")
                                    
                                    with st.expander("Lihat Kelas yang Tidak Mencukupi"):
                                        st.dataframe(insufficient_df,width='stretch')
                            
                            # Training/Validation Split for Sampling Data
                            st.divider()
                            st.subheader("D. Pembagian Data Training/Validasi")
                            
                            split_data = st.checkbox("Bagi data menjadi set training dan validasi", value=True, key="split_sampling")
                            
                            if split_data:
                                split_ratio = st.slider("Persentase data training:", min_value=0.5, max_value=0.9, value=0.7, step=0.05, key="split_ratio_sampling")
                                
                                if st.button("Bagi Data Training", type="primary", key="split_sampling_data"):
                                    try:
                                        if 'train_data_final_sampling' in st.session_state and st.session_state['train_data_final_sampling'] is not None:
                                            train_data = st.session_state['train_data_final_sampling']
                                            
                                            # Perform split
                                            train_final, valid_final = SplitTrainData.SplitProcess(
                                                train_data,
                                                TrainSplitPct=split_ratio,
                                                random_state=123
                                            )
                                            
                                            st.session_state['train_final_sampling'] = train_final
                                            st.session_state['valid_final_sampling'] = valid_final
                                            st.session_state['split_completed_sampling'] = True
                                            
                                            # Display split statistics
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Sampel Training", len(train_final))
                                            with col2:
                                                st.metric("Sampel Validasi", len(valid_final))
                                            with col3:
                                                st.metric("Total Sampel", len(train_data))
                                            
                                            st.success("Pembagian data berhasil diselesaikan!")
                                            
                                            # Show preview of split data
                                            with st.expander("Preview Data yang Dibagi"):
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.markdown("**Preview Data Training:**")
                                                    st.dataframe(train_final.head())
                                                with col2:
                                                    st.markdown("**Preview Data Validasi:**")
                                                    st.dataframe(valid_final.head())
                                                    
                                    except Exception as e:
                                        st.error(f"Error membagi data: {e}")
                            else:
                                # If not splitting, use all data as training
                                if 'train_data_final_sampling' in st.session_state and st.session_state['train_data_final_sampling'] is not None:
                                    st.session_state['train_final_sampling'] = st.session_state['train_data_final_sampling']
                                    st.session_state['valid_final_sampling'] = None
                                    st.session_state['split_completed_sampling'] = True
                                    st.info("Menggunakan seluruh dataset sebagai data training.")
                else:
                    st.error("AOI tidak tersedia untuk membuat peta sampling")
                    
            except Exception as e:
                st.error(f"Error membuat peta sampling: {e}")
                st.warning("Tidak dapat membuat peta interaktif. Silakan gunakan input manual.")
        else:
            st.error("Skema klasifikasi tidak tersedia. Silakan jalankan Modul 2 terlebih dahulu.")


# Navigation
st.divider()
st.subheader("Navigasi Modul")

col1, col2 = st.columns(2)

with col1:
    if st.button("⬅️ Kembali ke Modul 2: Skema Klasifikasi",width='stretch'):
        st.switch_page("pages/2_Module_2_Classification_scheme.py")

with col2:
    # Check if training data is ready from any source
    training_ready = (st.session_state.get('data_processed_ref', False) or 
                     st.session_state.get('data_processed_upload', False) or 
                     st.session_state.get('data_processed_sampling', False))
    
    if training_ready:
        if st.button("➡️ Lanjut ke Modul 4: Klasifikasi", type="primary",width='stretch'):
            st.switch_page("pages/4_Module_4_Analyze_ROI.py")
    else:
        st.button("🔒 Selesaikan Data Training Dulu", disabled=True,width='stretch',
                 help="Silakan proses data training sebelum melanjutkan")

# Status indicator
if training_ready:
    # Determine which data source was used and get sample count
    if st.session_state.get('data_processed_ref', False):
        total_samples = len(st.session_state.get('train_data_final_ref', []))
        data_source = "Referensi"
    elif st.session_state.get('data_processed_upload', False):
        total_samples = len(st.session_state.get('train_data_final_upload', []))
        data_source = "Upload"
    elif st.session_state.get('data_processed_sampling', False):
        total_samples = len(st.session_state.get('train_data_final_sampling', []))
        data_source = "Sampling"
    else:
        total_samples = 0
        data_source = "Unknown"
    
    st.success(f"✅ Data training siap dengan {total_samples} sampel ({data_source})")
else:
    st.info("Selesaikan pengumpulan dan pemrosesan data training untuk melanjutkan")