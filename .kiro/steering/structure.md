# Project Structure

## Directory Organization

```
├── Home.py                    # Main entry point and landing page
├── pages/                     # Streamlit multi-page modules
│   ├── 1_Module_1_Generate_Image_Mosaic.py
│   ├── 2_Module_2_Classification_scheme.py
│   ├── 3_Module_4_Analyze_ROI.py
│   ├── 4_Module_6_Classification and LULC Creation.py
│   └── 5_Module_7_Thematic_Accuracy.py
├── src/                       # Core processing logic
│   ├── module_helpers.py      # Shared utilities and helper functions
│   ├── src_modul_1.py        # Module 1 backend (Image collection search and preprocess)
│   ├── src_modul_2.py        # Module 2 backend (classification scheme)
│   ├── src_modul_4.py        # Module 4 backend (ROI and separability analysis)
│   ├── src_modul_4_part2.py  # Module 4 extended functionality (plot and visualization)
│   ├── src_modul_6.py        # Module 6 backend (LULC creation)
│   └── src_modul_7.py        # Module 7 backend (accuracy assessment)
├── logos/                     # Application branding assets
├── .streamlit/               # Streamlit configuration
│   └── secrets.toml          # Authentication credentials
├── notebooks/                # Jupyter notebooks (development/testing)
├── requirements.txt          # Python dependencies
└── packages.txt             # System dependencies
```

## Naming Conventions

### Files
- **Pages**: Numbered prefix for ordering (1_, 2_, etc.) + descriptive name
- **Source modules**: `src_modul_X.py` pattern matching page numbers
- **Classes**: PascalCase (e.g., `Reflectance_Data`, `EE_converter`)
- **Functions**: snake_case with descriptive names

### Session State Variables

#### Module 1 - Image Mosaic Generation
- `st.session_state.collection` - Earth Engine ImageCollection object from Landsat search
- `st.session_state.composite` - Processed image composite/mosaic for analysis
- `st.session_state.aoi` - Area of Interest as Earth Engine geometry
- `st.session_state.gdf` - GeoPandas DataFrame of uploaded AOI shapefile
- `st.session_state.export_tasks` - List of Earth Engine export task IDs
- `st.session_state.search_metadata` - Metadata from imagery search (sensor, dates, image count)
- `st.session_state.Image_metadata` - Detailed statistics about the image collection
- `st.session_state.AOI` - Area of Interest geometry for clipping operations
- `st.session_state.visualization` - Visualization parameters for map display

#### Module 2 - Classification Scheme
- `st.session_state.lulc_classes` - List of classification classes with ID, name, description, color
- `st.session_state.lulc_next_id` - Next available class ID for new classes
- `st.session_state.lulc_edit_mode` - Boolean flag for edit mode state
- `st.session_state.lulc_edit_idx` - Index of class being edited
- `st.session_state.csv_temp_classes` - Temporary storage for CSV-uploaded classes

#### Module 4 - ROI Analysis & Separability
- `st.session_state.training_data` - Earth Engine FeatureCollection of training ROI data
- `st.session_state.training_gdf` - GeoPandas DataFrame of uploaded training shapefile
- `st.session_state.selected_class_property` - Selected field name for class IDs
- `st.session_state.selected_class_name_property` - Selected field name for class names
- `st.session_state.analyzer` - Separability analysis object with computed statistics

#### Module 6 - LULC Classification
- `st.session_state.extracted_training_data` - Extracted spectral features for training
- `st.session_state.extracted_testing_data` - Extracted spectral features for testing
- `st.session_state.class_property` - Property name used for class identification
- `st.session_state.classification_result` - Generated land cover classification map
- `st.session_state.classification_mode` - Type of classification (Hard/Soft)
- `st.session_state.trained_model` - Trained machine learning model
- `st.session_state.classification_params` - Classification parameters (ntrees, etc.)
- `st.session_state.include_final_map` - Boolean for including final map in soft classification

#### Module 7 - Accuracy Assessment
- `st.session_state.validation_data` - Earth Engine FeatureCollection for validation
- `st.session_state.validation_gdf` - GeoPandas DataFrame of validation shapefile
- `st.session_state.accuracy_results` - Computed accuracy assessment results

#### System-wide
- `st.session_state.ee_initialized` - Boolean flag for Earth Engine initialization status

## Code Organization Patterns

### Page Structure
1. **Imports and initialization** - GEE init, required libraries
2. **Page configuration** - Title, layout, sidebar content
3. **Session state management** - Initialize required variables
4. **User input sections** - File uploads, parameter selection
5. **Processing logic** - Backend function calls
6. **Results display** - Maps, statistics, export options
7. **Navigation** - Module progression controls

### Backend Classes
- **Data retrieval classes**: Handle Earth Engine collections
- **Processing classes**: Image analysis and statistics
- **Helper classes**: Validation, conversion, utilities
- **Consistent logging**: Using Python logging module

### Error Handling
- Graceful degradation with multiple conversion attempts
- User-friendly error messages via Streamlit
- Validation at multiple stages (geometry, CRS, data quality)