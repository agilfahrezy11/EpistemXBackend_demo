import streamlit as st
from module_helpers import init_gee
init_gee()
from module_helpers import shapefile_validator, EE_converter
import ee
import pandas as pd
import geopandas as gpd

#Page configuration
st.set_page_config(
    page_title="Accuracy Assessment",
    page_icon="logos\logo_epistem_crop.png",
    layout="wide"
)
#Set the page title (for the canvas)
st.title("Thematic Accuracy Assessment")
st.divider()