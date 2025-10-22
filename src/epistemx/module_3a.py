import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape, Polygon, MultiPolygon
import ee
from datetime import datetime
import os
import folium
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display
from shapely.geometry import mapping
import random
import geemap
import time
import json
from typing import Any, Dict, List, Optional
from shapely.geometry import Point
import ipywidgets as widgets
from ipywidgets import Button, Output, VBox, HBox, Dropdown, Label, HTML
from ipyleaflet import Map, Marker, basemaps, LayersControl, GeoJSON, DivIcon

# --- System response 3.1 ---
class InputCheck:
    """
    System respons 3.1: Prequisite check
    """

    def ValidateVariable(*variable_names):
        """
        Validate that all specified variable names exist in the namespace.
        
        Args:
            *variable_names: Variable names as strings
        
        Usage:
            validate_variables('LULCTable', 'ClippedImage', 'AOI')
        """
        print("=== Validating Required Variables ===")
        
        missing_variables = []
        
        for var_name in variable_names:
            try:
                var_value = eval(var_name)
                print(f"{var_name}: EXISTS (type: {type(var_value)})")
            except NameError:
                print(f"{var_name}: NOT DEFINED")
                missing_variables.append(var_name)
            except Exception as e:
                print(f"{var_name}: ERROR - {e}")
                missing_variables.append(var_name)
        
        if missing_variables:
            error_msg = f"ERROR: Missing required variables: {', '.join(missing_variables)}. Execution stopped."
            print("\n" + "!" * 60)
            print(error_msg)
            print("!" * 60)
            raise NameError(error_msg)
        else:
            print("\nAll required variables are present. Continuing execution...")
            return True

# --- System response 3.2.a ---
class SyncTrainData:
    """
    Module 3: Syncronize input data training with the defined class
    Validates training data against land cover classes, AOI, and sample availibity
    """
    
    # --- System response 3.2.a ---

    def LoadTrainData(landcover_df, aoi_geometry, training_shp_path=None, training_ee_path=None):
        """
        Load training data from shapefile or Earth Engine asset
        
        Args:
            landcover_df: DataFrame from Module 2 with land cover classes
            aoi_geometry: ee.Geometry or GeoDataFrame representing the AOI
            training_shp_path: Path to shapefile training data
            training_ee_path: Earth Engine asset path for training data
        
        Returns:
            Dictionary with training_data and validation_results
        """
        validation_results = {
            'total_points': 0,
            'valid_points': 0,
            'points_after_class_filter': 0,
            'invalid_classes': [],
            'outside_aoi': [],
            'insufficient_samples': [],
            'warnings': []
        }
        
        if training_shp_path:
            training_data = gpd.read_file(training_shp_path)
            
        elif training_ee_path:
            ee_fc = ee.FeatureCollection(training_ee_path)
            
            # Convert to GeoDataFrame
            features = ee_fc.getInfo()['features']
            geometries = []
            properties_list = []
            
            for feature in features:
                geom_dict = feature['geometry']
                geom = shape(geom_dict)
                
                if geom.geom_type == 'MultiPoint':
                    for point in geom.geoms:
                        geometries.append(point)
                        properties_list.append(feature['properties'])
                elif geom.geom_type == 'Polygon':
                    geometries.append(geom.centroid)
                    properties_list.append(feature['properties'])
                elif geom.geom_type == 'Point':
                    geometries.append(geom)
                    properties_list.append(feature['properties'])
                else:
                    print(f"Warning: Skipping unsupported geometry type: {geom.geom_type}")
            
            training_data = gpd.GeoDataFrame(
                properties_list, 
                geometry=geometries,
                crs='EPSG:4326'
            )
        else:
            raise ValueError("Please provide either training_shp_path or training_ee_path")
        
        validation_results['total_points'] = len(training_data)
        
        return {
            'training_data': training_data,
            'landcover_df': landcover_df,
            'aoi_geometry': aoi_geometry,
            'validation_results': validation_results
        }

    def SetClassField(data_dict, field_name):
        """
        Define which field contains the land cover class
        
        Args:
            data_dict: Dictionary containing training data and validation results
            field_name: Name of the field containing class information
        """
        training_data = data_dict['training_data']
        
        if field_name not in training_data.columns:
            available_fields = list(training_data.columns)
            raise ValueError(f"Field '{field_name}' not found. Available fields: {available_fields}")
        
        data_dict['class_field'] = field_name
        
        return data_dict

    def ValidClass(data_dict):
        """
        Check if training data classes match Module 2 land cover classes
        
        Args:
            data_dict: Dictionary containing training data, landcover_df, and validation_results
        """
        training_data = data_dict['training_data']
        landcover_df = data_dict['landcover_df']
        class_field = data_dict['class_field']
        validation_results = data_dict['validation_results']
        
        # Get valid classes from Module 2 (second column)
        valid_classes = set(landcover_df.iloc[:, 1].values)
        
        # Get classes from training data
        training_classes = set(training_data[class_field].unique())
        
        # Find mismatches
        invalid_classes = training_classes - valid_classes
        
        if invalid_classes:
            for invalid_class in invalid_classes:
                count = len(training_data[training_data[class_field] == invalid_class])
                validation_results['invalid_classes'].append({
                    'class': invalid_class,
                    'count': count
                })
            
            # Filter out invalid classes
            original_count = len(training_data)
            training_data = training_data[training_data[class_field].isin(valid_classes)].copy()
            filtered_count = len(training_data)
            removed_count = original_count - filtered_count
            
            validation_results['warnings'].append(
                f"Removed {removed_count} points with invalid classes not in Module 2 definition"
            )
        
        validation_results['points_after_class_filter'] = len(training_data)
        data_dict['training_data'] = training_data
        
        return data_dict

    def CheckSufficiency(data_dict, min_samples=20):
        """
        Check if each class has sufficient training samples
        
        Args:
            data_dict: Dictionary containing training data, landcover_df, and validation_results
            min_samples: Minimum number of samples required per class
        """
        training_data = data_dict['training_data']
        landcover_df = data_dict['landcover_df']
        class_field = data_dict['class_field']
        validation_results = data_dict['validation_results']
        
        class_counts = training_data[class_field].value_counts()
        insufficient_classes = []
        
        for idx, row in landcover_df.iterrows():
            class_name = row.iloc[1]  # Second column for class name
            
            if class_name in class_counts:
                count = class_counts[class_name]
                if count < min_samples:
                    insufficient_classes.append({
                        'class': class_name,
                        'count': count,
                        'needed': min_samples - count
                    })
                    validation_results['insufficient_samples'].append({
                        'class': class_name,
                        'count': count
                    })
        
        if insufficient_classes:
            validation_results['warnings'].append(
                f"Found {len(insufficient_classes)} class(es) with insufficient samples (< {min_samples})"
            )
        
        # Check zero samples
        zero_sample_classes = []
        for idx, row in landcover_df.iterrows():
            class_name = row.iloc[1]
            if class_name not in class_counts:
                zero_sample_classes.append(class_name)
        
        if zero_sample_classes:
            validation_results['warnings'].append(
                f"Found {len(zero_sample_classes)} class(es) with no training samples"
            )
        
        return data_dict

    def FilterTrainAoi(data_dict):
        """
        Filter training points that fall within AOI
        
        Args:
            data_dict: Dictionary containing training data, aoi_geometry, and validation_results
        """
        training_data = data_dict['training_data']
        aoi_geometry = data_dict['aoi_geometry']
        validation_results = data_dict['validation_results']
        
        if isinstance(aoi_geometry, ee.Geometry):
            aoi_info = aoi_geometry.getInfo()
            aoi_type = aoi_info['type']
            
            if aoi_type == 'Polygon':
                aoi_coords = aoi_info['coordinates']
                aoi_polygon = Polygon(aoi_coords[0])
            elif aoi_type == 'MultiPolygon':
                from shapely.geometry import MultiPolygon
                polygons = [Polygon(coords[0]) for coords in aoi_info['coordinates']]
                aoi_polygon = MultiPolygon(polygons)
            else:
                aoi_polygon = shape(aoi_info)
            
            aoi_gdf = gpd.GeoDataFrame([1], geometry=[aoi_polygon], crs='EPSG:4326')
        else:
            aoi_gdf = aoi_geometry
        
        if training_data.crs != aoi_gdf.crs:
            training_data = training_data.to_crs(aoi_gdf.crs)
        
        within_aoi = training_data.geometry.within(aoi_gdf.unary_union)
        points_inside = training_data[within_aoi].copy()
        points_outside = training_data[~within_aoi].copy()
        
        if len(points_outside) > 0:
            for idx, row in points_outside.iterrows():
                class_name = row[data_dict['class_field']] if 'class_field' in data_dict else 'N/A'
                coords = row.geometry.coords[0]
                outside_info = {
                    'index': idx,
                    'class': class_name,
                    'lon': coords[0],
                    'lat': coords[1]
                }
                validation_results['outside_aoi'].append(outside_info)
            
            validation_results['warnings'].append(
                f"{len(points_outside)} points outside AOI will be ignored"
            )
        
        validation_results['valid_points'] = len(points_inside)
        data_dict['training_data'] = points_inside
        data_dict['points_outside'] = points_outside
        
        return data_dict

    def TrainDataRaw(training_data, landcover_df, class_field):
        """
        Create a table showing training data distribution
        
        Args:
            training_data: DataFrame containing training data
            landcover_df: DataFrame containing land cover class definitions
            class_field: Name of the column containing class information
        
        Returns:
            table_df: pandas DataFrame containing all table data
            total_samples: Total number of training samples
            insufficient_df: DataFrame containing only insufficient classes (or None if none exist)
        """
        if training_data is None or len(training_data) == 0:
            return None, 0, None
        
        # Create distribution table
        class_counts = training_data[class_field].value_counts()
        total_valid = len(training_data)
        
        table_data = []
        for idx, row in landcover_df.iterrows():
            class_id = row.iloc[0]
            class_name = row.iloc[1]
            
            if class_name in class_counts:
                count = class_counts[class_name]
                percentage = (count / total_valid) * 100
                status = "Sufficient" if count >= 20 else "Insufficient"
            else:
                count, percentage, status = 0, 0, "No Samples"
            
            table_data.append({
                'ID': class_id,
                'LULC_class': class_name,
                'Sample_Count': count,
                'Percentage': percentage,
                'Status': status
            })
        
        table_df = pd.DataFrame(table_data)
        
        # Create insufficient classes DataFrame
        insufficient_df = table_df[table_df['Status'].isin(['Insufficient', 'No Samples'])].copy()
        if len(insufficient_df) == 0:
            insufficient_df = None
        
        return table_df, total_valid, insufficient_df
    
    def generate_report(self, output_path='modul-3_report.txt'):
        """
        Generate a text report of the validation results
        """
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_lines = [
            "="*70,
            "REPORT",
            "="*70,
            f"Generated: {timestamp}",
            "",
            "SUMMARY",
            "-"*70,
            f"Total training points loaded: {self.validation_results['total_points']}",
            f"Points after class filtering: {self.validation_results['points_after_class_filter']}",
            f"Valid points (inside AOI + valid classes): {self.validation_results['valid_points']}",
            f"Points outside AOI: {len(self.validation_results['outside_aoi'])}",
            f"Invalid classes found: {len(self.validation_results['invalid_classes'])}",
            f"Classes with insufficient samples: {len(self.validation_results['insufficient_samples'])}",
            "-"*70,
        ]
        
        # Use column order:
        # 0 = ID, 1 = Class Name, 2 = Color Palette
        # for idx, row in self.landcover_df.iterrows():
        #    report_lines.append(f"  {row.iloc[0]}. {row.iloc[1]} - {row.iloc[2]}")
        
        if self.training_data is not None and len(self.training_data) > 0:
            report_lines.extend([
                "",
                "VALID TRAINING DATA DISTRIBUTION (Module 2 Classes Only)",
                "-"*70,
            ])
            
            class_counts = self.training_data[self.class_field].value_counts()
            total_valid = len(self.training_data)
            
            report_lines.append(f"{'ID':<5} {'LULC_class':<30} {'Sample_Count':<15} {'Percentage':<12} {'Status':<20}")
            report_lines.append("-" * 82)
            
            for idx, row in self.landcover_df.iterrows():
                class_id = row.iloc[0]
                class_name = row.iloc[1]
                
                if class_name in class_counts:
                    count = class_counts[class_name]
                    percentage = (count / total_valid) * 100
                    status = "Sufficient" if count >= 20 else "Insufficient"
                else:
                    count, percentage, status = 0, 0, "No Samples"
                
                report_lines.append(
                    f"{class_id:<5} {class_name:<30} {count:<15} {percentage:>10.2f}%  {status:<20}"
                )
            
            report_lines.append("-" * 82)
            report_lines.append(f"{'TOTAL':<5} {'':<30} {total_valid:<15} {100.00:>10.2f}%")
        
        # Write report
        report_text = "\n".join(report_lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\nModul 3 report saved to: {output_path}")
        print(report_text)
        
        return report_text
    
    def get_valid_training_data(self):
        """
        Return validated training data
        """
        return self.training_data

# ----- System response 3.5 -----
class SplitTrainData:

    """
    """

    def SplitProcess(TrainDataRecap, TrainSplitPct=0.8, random_state=42):
        """
        Split training data into train and validation sets.
        
        Parameters:
        -----------
        TrainDataRecap : GeoDataFrame
            The input geodataframe containing training data with a 'kelas' column
        TrainSplitPct : float, optional (default=0.8)
            Percentage of data to use for training (0.0 to 1.0)
        random_state : int, optional (default=42)
            Random seed for reproducibility
        
        Returns:
        --------
        tuple
            (TrainDataFinal, ValidDataFinal) - Training and validation GeoDataFrames
        """
        from sklearn.model_selection import train_test_split
        
        # Perform train-test split with stratification if possible
        try:
            TrainDataFinal, ValidDataFinal = train_test_split(
                TrainDataRecap,
                train_size=TrainSplitPct,
                stratify=TrainDataRecap['kelas'],
                random_state=random_state
            )
        except ValueError:
            print("Stratified split not possible, using random split.")
            TrainDataFinal, ValidDataFinal = train_test_split(
                TrainDataRecap,
                train_size=TrainSplitPct,
                random_state=random_state
            )

        # Check and remove overlaps
        overlap = TrainDataFinal.index.intersection(ValidDataFinal.index)
        if len(overlap) > 0:
            print(f"Overlap detected ({len(overlap)} rows). Removing duplicates.")
            ValidDataFinal = ValidDataFinal.drop(index=overlap)
        
        return TrainDataFinal, ValidDataFinal

    def PlotTrainValidInteractive(TrainDataFinal, AOI, ValidDataFinal=None):
        """
        Plot training and optional validation data interactively using Folium.
        
        Parameters:
            TrainDataFinal (GeoDataFrame): Training point data.
            AOI (GeoDataFrame or ee.Geometry): AOI geometry.
            ValidDataFinal (GeoDataFrame, optional): Validation point data. Defaults to None.
        """
        
        # Convert AOI (if ee.Geometry) to GeoDataFrame 
        if not isinstance(AOI, gpd.GeoDataFrame):
            try:
                # If AOI is an ee.Geometry, convert to GeoJSON
                aoi_geojson = AOI.getInfo()
                AOI = gpd.GeoDataFrame.from_features([{
                    'geometry': aoi_geojson,
                    'properties': {}
                }], crs='EPSG:4326')
            except Exception as e:
                print("AOI could not be converted:", e)
                return

        # Get AOI center for map initialization
        bounds = AOI.total_bounds  # minx, miny, maxx, maxy
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

        # Initialize folium map 
        m = folium.Map(location=center, zoom_start=10, tiles='OpenStreetMap')

        # Add AOI outline
        folium.GeoJson(
            AOI,
            name="AOI Boundary",
            style_function=lambda x: {'color': 'blue', 'weight': 2, 'fillOpacity': 0}
        ).add_to(m)

        # Add training points
        for _, row in TrainDataFinal.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.6,
                popup="Training Point"
            ).add_to(m)

        # Add validation points (if available) 
        if isinstance(ValidDataFinal, gpd.GeoDataFrame) and not ValidDataFinal.empty:
            for _, row in ValidDataFinal.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    color='orange',
                    fill=True,
                    fill_color='orange',
                    fill_opacity=0.6,
                    popup="Validation Point"
                ).add_to(m)

        # Add legend manually
        legend_html = """
        <div style="
            position: fixed;
            bottom: 20px; left: 20px; width: 180px;
            background-color: white; z-index:9999; font-size:14px;
            border:2px solid grey; border-radius:5px; padding: 8px;">
        <b>Legend</b><br>
        <i style="background:green; width:10px; height:10px; float:left; margin-right:5px; opacity:0.6"></i> Training Data<br>
        <i style="background:orange; width:10px; height:10px; float:left; margin-right:5px; opacity:0.6"></i> Validation Data<br>
        <i style="background:blue; width:10px; height:10px; float:left; margin-right:5px; opacity:0.6"></i> AOI Boundary
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        folium.LayerControl().add_to(m)

        display(m)

# ----- System respons 3.3.a -----
class LULCSamplingTool:
    """
    LULC Sampling Tool using ipyleaflet for interactive point sampling.

    This class expects a pandas DataFrame containing class definitions and optionally
    an Earth Engine FeatureCollection (geometry) for AOI restriction.
    """

    def __init__(self, lulc_dataframe: pd.DataFrame, aoi_ee_featurecollection: Optional[Any] = None) -> None:
        """
        Initialize the sampling tool.

        Parameters:
            lulc_dataframe: pd.DataFrame containing at least ID and class/type and color
            aoi_ee_featurecollection: Optional Earth Engine FeatureCollection/Geometry for AOI
        """
        self.lulc_df = lulc_dataframe
        self.aoi_ee_featurecollection = aoi_ee_featurecollection
        self.aoi_geometry = None
        self.training_data: List[Dict[str, Any]] = []
        self.TrainDataSampling = pd.DataFrame(columns=['ID', 'LULC_Type', 'Points', 'Coordinates'])
        self.current_class: Optional[Dict[str, Any]] = None
        self.markers: List[Marker] = []
        self.marker_data_map: Dict[Any, Dict[str, Any]] = {}
        self.aoi_layer = None
        self.point_counter: Dict[str, int] = {}
        self.edit_mode: bool = False
        self.last_click_time: float = 0.0

        # Initialize UI and related components
        self.CreateUi()

        # Initialize point counter
        for idx, row in self.lulc_df.iterrows():
            self.point_counter[row['LULC_Type']] = 0

        # Load AOI if provided
        if self.aoi_ee_featurecollection is not None:
            self.LoadAoiFromEe()

        # Create map with dynamic center/zoom
        # If AOI not loaded, set defaults
        if not hasattr(self, 'map_center'):
            self.map_center = [0, 0]
        if not hasattr(self, 'zoom'):
            self.zoom = 10

        self.CreateMap()

    def LoadAoiFromEe(self) -> None:
        """
        Load AOI from Earth Engine FeatureCollection (or Geometry) and prepare map center and zoom.
        """
        try:
            aoi_geojson = self.aoi_ee_featurecollection.geometry().getInfo()
            self.aoi_gdf = gpd.GeoDataFrame.from_features([{
                "type": "Feature",
                "geometry": aoi_geojson,
                "properties": {}
            }], crs="EPSG:4326")

            # Use union_all() to merge geometries if needed
            self.aoi_geometry = self.aoi_gdf.union_all()

            bounds = self.aoi_gdf.bounds
            minx, miny, maxx, maxy = bounds.iloc[0]
            self.map_center = [(miny + maxy) / 2, (minx + maxx) / 2]

            width_deg = maxx - minx
            height_deg = maxy - miny
            max_dimension = max(width_deg, height_deg)

            if max_dimension > 20:
                self.zoom = 6
            elif max_dimension > 10:
                self.zoom = 7
            elif max_dimension > 5:
                self.zoom = 8
            elif max_dimension > 2:
                self.zoom = 9
            elif max_dimension > 1:
                self.zoom = 10
            elif max_dimension > 0.5:
                self.zoom = 11
            elif max_dimension > 0.2:
                self.zoom = 12
            elif max_dimension > 0.1:
                self.zoom = 13
            elif max_dimension > 0.05:
                self.zoom = 14
            else:
                self.zoom = 15

            with self.output:
                print("SUCCESS: AOI loaded from Earth Engine FeatureCollection")
                print(f"  - CRS: {self.aoi_gdf.crs}")
                try:
                    area_value = float(self.aoi_geometry.area)
                    print(f"  - Area: {area_value:.6f} square degrees")
                except Exception:
                    print("  - Area: unavailable")
                print(f"  - Map center: {self.map_center}")
                # print(f"  - Zoom level: {self.zoom} (showing entire AOI)")

        except Exception as e:
            with self.output:
                print(f"ERROR: Error loading AOI from Earth Engine: {str(e)}")
            self.aoi_geometry = None
            self.map_center = [0, 0]
            self.zoom = 10

    def CreateMap(self) -> None:
        """
        Create ipyleaflet map with computed center and zoom, add controls and interaction.
        """
        self.map = Map(
            center=self.map_center,
            zoom=self.zoom,
            basemap=basemaps.Esri.WorldImagery,
            scroll_wheel_zoom=True
        )

        self.map.add_control(LayersControl())

        if self.aoi_geometry is not None:
            self.AddAoiLayer()

        # Map interaction event
        self.map.on_interaction(self.HandleMapClick)

        # Add custom cursor styling
        self.AddCrosshairCursor()

    def AddCrosshairCursor(self) -> None:
        """
        Add CSS that sets the cursor to crosshair over the leaflet map container.
        """
        crosshair_style = """
        <style>
        .jupyter-widgets.widget-container.widget-box.widget-vbox .jupyter-widgets.widget-container.widget-box.widget-vbox {
            cursor: default !important;
        }
        .leaflet-container {
            cursor: crosshair !important;
        }
        </style>
        """
        display(HTML(crosshair_style))

    def AddAoiLayer(self) -> None:
        """
        Add AOI boundary layer (white outline, no fill) to the map.
        """
        try:
            aoi_geojson = json.loads(self.aoi_gdf.to_json())
            self.aoi_layer = GeoJSON(
                data=aoi_geojson,
                style={
                    'color': 'white',
                    'weight': 3,
                    'fillColor': 'white',
                    'fillOpacity': 0.0,
                    'opacity': 1.0,
                    'dashArray': '5, 5'
                },
                name="AOI Boundary"
            )
            self.map.add_layer(self.aoi_layer)
            with self.output:
                print("SUCCESS: AOI boundary layer added to map")
        except Exception as e:
            with self.output:
                print(f"ERROR: Error adding AOI layer to map: {str(e)}")

    def IsPointInAoi(self, lat: float, lon: float) -> bool:
        """
        Check whether a geographic point (lat, lon) is within the AOI boundary.
        Returns True if no AOI is set (points allowed everywhere).
        """
        if self.aoi_geometry is None:
            return True
        point = Point(lon, lat)
        return self.aoi_geometry.contains(point)

    def HandleMapClick(self, **kwargs) -> None:
        """
        Handle click interactions from the ipyleaflet map.
        Expects kwargs containing 'type' and 'coordinates' for click events.
        """
        if kwargs.get('type') == 'click':
            if self.current_class is None:
                with self.output:
                    print("WARNING: Please select a class first!")
                return

            coords = kwargs.get('coordinates')
            if coords:
                lat, lon = coords

                if not self.IsPointInAoi(lat, lon):
                    with self.output:
                        print(f"ERROR: Point at ({lat:.6f}, {lon:.6f}) is outside AOI boundary! Point rejected.")
                    return

                self.AddPointMarker(lat, lon)

    def AddPointMarker(self, lat: float, lon: float) -> None:
        """
        Add a marker to the map for the currently selected class and store it in training data.
        """
        color = self.current_class.get('color', '#0000FF')

        marker = self.CreateCustomMarker(lat, lon, color)
        marker.draggable = self.edit_mode

        marker_data = {
            'latitude': lat,
            'longitude': lon,
            'class_id': self.current_class['id'],
            'class_type': self.current_class['type'],
            'color': color
        }
        self.marker_data_map[marker] = marker_data

        def _HandleMove(event: Any) -> None:
            if not self.edit_mode:
                return

            # traitlets observe passes change dict; ipyleaflet may pass (name, old, new) etc.
            # Try to get new coordinates robustly
            new_lat, new_lon = None, None
            try:
                # If event is a dict from traitlets
                if isinstance(event, dict) and 'new' in event:
                    new_lat, new_lon = event['new']
                elif hasattr(event, 'new'):
                    new_lat, new_lon = event.new
                else:
                    new_lat, new_lon = marker.location
            except Exception:
                new_lat, new_lon = marker.location

            old_lat = self.marker_data_map[marker]['latitude']
            old_lon = self.marker_data_map[marker]['longitude']

            if not self.IsPointInAoi(new_lat, new_lon):
                with self.output:
                    print(f"ERROR: Cannot move point to ({new_lat:.6f}, {new_lon:.6f}) - outside AOI boundary!")
                marker.location = (old_lat, old_lon)
                return

            # Update internal state
            self.marker_data_map[marker]['latitude'] = new_lat
            self.marker_data_map[marker]['longitude'] = new_lon

            for i, point in enumerate(self.training_data):
                if (point['latitude'] == old_lat and point['longitude'] == old_lon and
                        point['class_id'] == self.current_class['id']):
                    self.training_data[i]['latitude'] = new_lat
                    self.training_data[i]['longitude'] = new_lon
                    break

            with self.output:
                print(f"SUCCESS: Point moved from ({old_lat:.6f}, {old_lon:.6f}) to ({new_lat:.6f}, {new_lon:.6f})")

            self.UpdateTrainDataSampling()
            self.UpdateTableDisplay()

        marker.observe(_HandleMove, names=['location'])

        def _HandleClick(**ev_kwargs: Any) -> None:
            current_time = time.time()
            time_diff = current_time - self.last_click_time
            if time_diff < 0.5 and self.edit_mode:
                self.RemovePoint(marker)
            self.last_click_time = current_time

        marker.on_click(_HandleClick)

        self.map.add_layer(marker)
        self.markers.append(marker)
        self.training_data.append(marker_data)

        class_type = self.current_class['type']
        self.point_counter[class_type] = self.point_counter.get(class_type, 0) + 1

        with self.output:
            print(f"SUCCESS: Point added at ({lat:.6f}, {lon:.6f}) for class: {class_type}")

        self.UpdateStatistics()
        self.UpdateTrainDataSampling()
        self.UpdateTableDisplay()

    def CreateCustomMarker(self, lat: float, lon: float, hex_color: str) -> Marker:
        """
        Create a custom DivIcon marker with exact hex color and return an ipyleaflet.Marker.
        """
        icon_html = f"""
        <div style="
            background-color: {hex_color};
            width: 20px;
            height: 20px;
            border: 2px solid white;
            border-radius: 50%;
            box-shadow: 0 0 5px rgba(0,0,0,0.5);
            cursor: pointer;
        "></div>
        """

        marker = Marker(
            location=(lat, lon),
            draggable=self.edit_mode,
            icon=DivIcon(
                icon_size=(20, 20),
                icon_anchor=(10, 10),
                html=icon_html
            )
        )
        return marker

    def RemovePoint(self, marker: Marker) -> None:
        """
        Remove a specified marker from the map and from stored training data.
        """
        if marker in self.markers:
            self.map.remove_layer(marker)
            self.markers.remove(marker)

            marker_data = self.marker_data_map.get(marker, {})
            lat = marker_data.get('latitude', 0)
            lon = marker_data.get('longitude', 0)
            class_type = marker_data.get('class_type', 'Unknown')

            self.training_data = [
                point for point in self.training_data
                if not (point['latitude'] == lat and point['longitude'] == lon and point['class_type'] == class_type)
            ]

            if marker in self.marker_data_map:
                del self.marker_data_map[marker]

            if class_type in self.point_counter:
                self.point_counter[class_type] = max(0, self.point_counter[class_type] - 1)

            with self.output:
                print(f"SUCCESS: Point removed at ({lat:.6f}, {lon:.6f}) for class: {class_type}")

            self.UpdateStatistics()
            self.UpdateTrainDataSampling()
            self.UpdateTableDisplay()

    def ToggleEditMode(self, b: Optional[Any] = None) -> None:
        """
        Toggle edit mode on/off. In edit mode markers are draggable and double-click removes them.
        """
        self.edit_mode = not self.edit_mode

        for marker in self.markers:
            marker.draggable = self.edit_mode

        if self.edit_mode:
            self.edit_btn.button_style = 'warning'
            self.edit_btn.description = 'Edit Mode: ON'
            with self.output:
                print("INFO: Edit Mode: ON - Double-click points to remove them, drag to reposition")
        else:
            self.edit_btn.button_style = ''
            self.edit_btn.description = 'Edit Mode: OFF'
            with self.output:
                print("INFO: Edit Mode: OFF - Points are now locked")

    def CreateUi(self) -> None:
        """
        Create the user interface widgets and prepare output areas.
        """
        id_col = 'ID'
        type_col = None
        color_col = None

        for col in self.lulc_df.columns:
            col_lower = col.lower()
            if 'type' in col_lower or 'class' in col_lower:
                type_col = col
            elif 'color' in col_lower or 'palette' in col_lower:
                color_col = col

        if type_col is None:
            type_col = self.lulc_df.columns[1]
        if color_col is None:
            color_col = self.lulc_df.columns[2] if len(self.lulc_df.columns) > 2 else self.lulc_df.columns[1]

        self.id_col = id_col
        self.type_col = type_col
        self.color_col = color_col

        class_options = [(f"{row[self.id_col]}: {row[self.type_col]}", idx)
                         for idx, row in self.lulc_df.iterrows()]

        self.class_dropdown = Dropdown(
            options=class_options,
            description='Select Class:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )

        self.class_info = Label(value="Select a class to start sampling")
        aoi_status = "Loaded" if self.aoi_ee_featurecollection is not None else "Not provided"
        # self.aoi_info = Label(value=f"AOI Status: {aoi_status}")

        self.update_class_btn = Button(
            description='Set Active Class',
            button_style='success',
            layout=widgets.Layout(width='150px')
        )
        self.update_class_btn.on_click(self.OnClassSelect)

        self.save_btn = Button(
            description='Update Data',
            button_style='primary',
            layout=widgets.Layout(width='150px')
        )
        self.save_btn.on_click(self.SaveTrainingData)

        self.clear_btn = Button(
            description='Clear All Points',
            button_style='danger',
            layout=widgets.Layout(width='150px')
        )
        self.clear_btn.on_click(self.ClearData)

        self.edit_btn = Button(
            description='Edit Mode: OFF',
            button_style='',
            layout=widgets.Layout(width='150px')
        )
        self.edit_btn.on_click(self.ToggleEditMode)

        self.export_btn = Button(
            description='Export to Shapefile',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        self.export_btn.on_click(self.ExportToShapefile)

        self.output = Output()
        self.stats_output = Output()
        self.table_output = Output()

    def OnClassSelect(self, b: Optional[Any] = None) -> None:
        """
        Handle class selection from dropdown and set the active class.
        """
        if self.class_dropdown.value is None:
            with self.output:
                print("WARNING: Please select a class from the dropdown!")
            return

        idx = self.class_dropdown.value
        row = self.lulc_df.iloc[idx]

        self.current_class = {
            'id': row[self.id_col],
            'type': row[self.type_col],
            'color': row[self.color_col]
        }
        self.class_info.value = f"Active Class: {row[self.id_col]} - {row[self.type_col]} (Color: {row[self.color_col]})"

        with self.output:
            print(f"SUCCESS: Active class set to: {row[self.type_col]}")

    def SaveTrainingData(self, b: Optional[Any] = None) -> None:
        """
        Update TrainDataSampling variable with current training data and display info.
        """
        if not self.training_data:
            with self.output:
                print("WARNING: No training data to save!")
            return

        self.UpdateTrainDataSampling()

        with self.output:
            print("SUCCESS: Training data updated successfully!")
            print(f"  - Total samples: {len(self.training_data)}")
            print("  - Data stored in: tool.TrainDataSampling")

        self.UpdateTableDisplay()

    def UpdateTrainDataSampling(self) -> None:
        """
        Build/refresh the TrainDataSampling DataFrame that summarizes points per class.
        """
        summary_data: List[Dict[str, Any]] = []

        if not self.training_data:
            for idx, row in self.lulc_df.iterrows():
                summary_data.append({
                    'ID': int(row[self.id_col]),
                    'LULC_Type': row[self.type_col],
                    'Points': 0,
                    'Coordinates': ''
                })
            self.TrainDataSampling = pd.DataFrame(summary_data)
            return

        df = pd.DataFrame(self.training_data)
        all_classes = set(self.lulc_df[self.id_col])
        sampled_classes = set(df['class_id'].unique())

        for class_id in all_classes:
            class_row = self.lulc_df[self.lulc_df[self.id_col] == class_id].iloc[0]
            if class_id in sampled_classes:
                group = df[df['class_id'] == class_id]
                coords = list(zip(group['latitude'], group['longitude']))
                coords_str = '; '.join([f"({lat:.6f}, {lon:.6f})" for lat, lon in coords])
                summary_data.append({
                    'ID': int(class_id),
                    'LULC_Type': class_row[self.type_col],
                    'Points': len(group),
                    'Coordinates': coords_str
                })
            else:
                summary_data.append({
                    'ID': int(class_id),
                    'LULC_Type': class_row[self.type_col],
                    'Points': 0,
                    'Coordinates': ''
                })

        self.TrainDataSampling = pd.DataFrame(summary_data).sort_values('ID')

    def ClearData(self, b: Optional[Any] = None) -> None:
        """
        Clear all training data and remove markers from the map.
        """
        for marker in self.markers:
            try:
                self.map.remove_layer(marker)
            except Exception:
                pass

        self.markers = []
        self.training_data = []
        self.marker_data_map = {}

        for key in self.point_counter:
            self.point_counter[key] = 0

        self.edit_mode = False
        self.edit_btn.button_style = ''
        self.edit_btn.description = 'Edit Mode: OFF'

        with self.output:
            print("SUCCESS: All training data and markers cleared!")

        self.UpdateTrainDataSampling()
        self.UpdateStatistics()
        self.UpdateTableDisplay()

    def UpdateStatistics(self) -> None:
        """
        Show a simple statistics summary in the stats_output area.
        """
        with self.stats_output:
            self.stats_output.clear_output()
            if not self.training_data:
                print("INFO: No samples collected yet.")
                return

            print("INFO: === Training Data Statistics ===")
            print(f"Total points: {len(self.training_data)}")
            # print("\nPoints per class:")
            # for class_type, count in self.point_counter.items():
            #     if count > 0:
            #         print(f"  {class_type}: {count}")

    def UpdateTableDisplay(self) -> None:
        """
        Display the TrainDataSampling DataFrame in a styled format in the table_output area.
        """
        with self.table_output:
            self.table_output.clear_output(wait=True)

            if self.TrainDataSampling.empty:
                print("INFO: No data to display.")
            else:
                display_df = self.TrainDataSampling.copy()
                styled_df = display_df.style.set_properties(**{
                    'background-color': '#f8f9fa',
                    'border': '1px solid #dee2e6',
                    'padding': '8px',
                    'text-align': 'left'
                }).set_table_styles([{
                    'selector': 'thead th',
                    'props': [('background-color', '#007bff'),
                             ('color', 'white'),
                             ('font-weight', 'bold'),
                             ('padding', '12px')]
                }])

                print("INFO: === Training Data Summary (TrainDataSampling) ===")
                display(styled_df)

    def ExportToShapefile(self, b: Optional[Any] = None):
        """
        Export current training data to a shapefile in the output/ directory.
        Returns the GeoDataFrame if successful, otherwise None.
        """
        if not self.training_data:
            with self.output:
                print("WARNING: No training data to export!")
            return None

        try:
            geometries = [Point(item['longitude'], item['latitude']) for item in self.training_data]
            attributes_df = pd.DataFrame([
                {
                    'class_id': item['class_id'],
                    'class_type': item['class_type'],
                    'color': item['color']
                } for item in self.training_data
            ])

            gdf = gpd.GeoDataFrame(
                attributes_df,
                geometry=geometries,
                crs="EPSG:4326"
            )

            os.makedirs('output', exist_ok=True)
            shapefile_path = 'output/training_data_points.shp'
            gdf.to_file(shapefile_path, driver='ESRI Shapefile')

            verified_gdf = gpd.read_file(shapefile_path)

            with self.output:
                print("SUCCESS: Training data exported to shapefile!")
                print(f"  - File: {shapefile_path}")
                print(f"  - Total features: {len(gdf)}")
                print(f"  - CRS: {gdf.crs}")
                print(f"  - Coordinate order: (longitude, latitude)")
                try:
                    bounds = verified_gdf.total_bounds
                    print(f"  - Bounds: {bounds}")
                except Exception:
                    print("  - Bounds: unavailable")
                print("\nShapefile components created:")
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    file_path = f'output/training_data_points{ext}'
                    if os.path.exists(file_path):
                        print(f"  - {file_path}")

            return gdf

        except Exception as e:
            with self.output:
                print(f"ERROR: Error exporting to shapefile: {str(e)}")
            return None

    def Display(self) -> None:
        """
        Render the UI controls, the map, and the output areas in the notebook.
        """
        instructions = HTML(
            """
            <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                <h3 style='margin-top: 0;'>LULC Point Sampling Tool</h3>
                <ol>
                    <li>Select a LULC class from the dropdown menu</li>
                    <li>Click <b>'Set Active Class'</b> button</li>
                    <li><b>Click on the map</b> to add points for the selected class</li>
                    <li>Each class will have points in different colors</li>
                    <li>Click <b>'Update Data'</b> to update the summary table</li>
                    <li>Use <b>'Clear All Points'</b> to start over</li>
                    <li>Toggle <b>'Edit Mode'</b> to remove points (double-click) or reposition (drag)</li>
                    <li>Click <b>'Export to Shapefile'</b> to save as GIS vector data</li>
                </ol>
                <p><b>Note:</b> Hover over the map to activate crosshair cursor for precise point placement</p>
                <p><b>AOI Restriction:</b> Points can only be placed within the AOI boundary (white outline)</p>
            </div>
            """
        )

        controls_top = VBox([
            instructions,
            Label(value="=== Class Selection ==="),
            self.class_dropdown,
            self.update_class_btn,
            self.class_info,
            self.aoi_info,
            Label(value=""),
            HBox([self.save_btn, self.clear_btn, self.edit_btn, self.export_btn]),
            self.stats_output,
            self.table_output
        ])

        display(controls_top)
        display(self.map)
        display(self.output)

        self.UpdateTrainDataSampling()
        self.UpdateTableDisplay()