import streamlit as st
import geemap.foliumap as geemap
from streamlit_folium import st_folium
import ee
import geopandas as gpd
import pandas as pd
ee.Authenticate()
ee.Initialize()
import folium
from folium.plugins import Draw

st.title("Digitize Training Data")

m = folium.Map(location=[-2, 118], zoom_start=5)
draw = Draw(export=True)
draw.add_to(m)

output = st_folium(m, width=700, height=500)

if output and "all_drawings" in output and output["all_drawings"]:
    st.write("User drawings (GeoJSON):")
    st.json(output["all_drawings"])

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(output["all_drawings"]["features"])
    st.write("Converted to GeoDataFrame:", gdf)
