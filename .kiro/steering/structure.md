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
│   ├── src_modul_1.py        # Module 1 backend (image collection)
│   ├── src_modul_2.py        # Module 2 backend (classification scheme)
│   ├── src_modul_4.py        # Module 4 backend (ROI and separability analysis)
│   ├── src_modul_4_part2.py  # Module 4 extended functionality
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
- `st.session_state.collection` - Image collection objects
- `st.session_state.composite` - Processed image composites
- `st.session_state.aoi` - Area of Interest geometry
- `st.session_state.gdf` - GeoPandas DataFrames
- `st.session_state.export_tasks` - Earth Engine export tracking

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