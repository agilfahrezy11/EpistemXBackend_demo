---
inclusion: always
---

# Technology Stack & Development Guidelines

## Core Technologies
- **Streamlit**: Multi-page web application framework - use `st.session_state` for data persistence across modules
- **Google Earth Engine (GEE)**: Primary satellite imagery platform - always initialize with `init_gee()` before GEE operations
- **Python 3.x**: Core language - follow PEP 8 conventions

## Essential Libraries
- **ee**: Google Earth Engine Python API - use for all satellite data operations
- **geemap/leafmap**: Interactive mapping - prefer geemap for GEE integration
- **geopandas**: Geospatial data handling - validate geometries before GEE conversion
- **streamlit-folium**: Map visualization in Streamlit
- **pandas/numpy**: Data manipulation - use for statistical analysis and data processing
- **shapely**: Geometric operations - validate coordinates before processing

## Authentication Requirements
- **Earth Engine**: Service account authentication via `st.secrets["earthengine"]`
- Always check `st.session_state.ee_initialized` before GEE operations
- Handle authentication failures gracefully with user-friendly error messages

## Code Style Guidelines

### Class Naming
- Use PascalCase with descriptive names: `Reflectance_Data`, `EE_converter`
- Group related functionality in classes (e.g., `sample_quality` for separability analysis)

### Function Naming
- Use snake_case: `validate_and_fix_geometry`, `extract_spectral_values`
- Prefix private methods with underscore: `_fix_crs`, `_clean_geometries`

### Session State Management
- Use descriptive keys: `st.session_state.composite`, `st.session_state.training_data`
- Initialize session state variables at page start
- Clear temporary variables when no longer needed

### Error Handling
- Implement graceful degradation with multiple conversion attempts
- Use try-catch blocks for GEE operations and file I/O
- Provide user-friendly error messages via `st.error()`
- Log technical details for debugging

### Geospatial Data Processing
- Always validate geometries before GEE conversion using `shapefile_validator` class
- Handle CRS transformations explicitly
- Validate coordinate ranges and vertex counts
- Use `EE_converter` class for consistent GDF to GEE conversions

## Development Workflow
```bash
# Local development
streamlit run Home.py 
python -m streamlit run Home.py 

# Install dependencies
pip install -r requirements.txt
```

## Architecture Patterns
- **Page Structure**: Import → Config → Session State → Input → Processing → Display → Navigation
- **Backend Classes**: Separate data retrieval, processing, and helper functionality
- **Module Dependencies**: Check prerequisites (composite, training data) before processing
- **Export Operations**: Use Earth Engine Task Manager for large exports to Google Drive