# Technology Stack

## Framework & Platform
- **Streamlit**: Multi-page web application framework
- **Google Earth Engine (GEE)**: Satellite imagery processing and analysis
- **Python 3.x**: Core programming language

## Key Libraries
- **geemap**: Earth Engine integration with interactive mapping
- **geopandas**: Geospatial data manipulation and analysis
- **leafmap**: Interactive mapping with Folium backend
- **streamlit-folium**: Streamlit-Folium integration
- **ee**: Google Earth Engine Python API
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **shapely**: Geometric operations
- **owslib**: OGC web services client

## System Dependencies
- **GDAL**: Geospatial Data Abstraction Library
- **gdal-bin**: GDAL command line utilities
- **libgdal-dev**: GDAL development libraries

## Authentication & Deployment
- **Google Service Account**: Earth Engine authentication via `st.secrets["earthengine"]`
- **Streamlit Cloud**: Deployment platform
- **Google Drive**: Export destination for processed imagery

## Common Commands

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run Home.py

# Check Streamlit version
streamlit --version
```

### Earth Engine Setup
- Requires service account credentials in `.streamlit/secrets.toml`
- Authentication handled automatically via `init_gee()` function
- Export tasks monitored via Earth Engine Task Manager

### File Structure
- Main entry point: `Home.py`
- Module pages: `pages/` directory with numbered prefixes
- Source code: `src/` directory with module helpers and processing logic
- Assets: `logos/` directory for branding