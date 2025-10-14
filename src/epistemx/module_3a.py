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