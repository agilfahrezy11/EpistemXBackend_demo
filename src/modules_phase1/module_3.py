import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ee.Initialize()

# Step: 3.3 Verifying the Input Reference Data
# Sub-Step: 3.3.1 Verifying the Input Reference Data

class sample_quality:
    """
    class which contain several functions used to conduct training data/sample analysis 
    
    """
    def __init__(self, training_data, image, class_property, region, class_name_property=None):
        """
        Initialize the tools for conducting the analysis
        Args:
            training_data: ee.FeatureCollection - Training polygons/points
            image: ee.Image - The image to extract spectral values
            class_property: str - Property/columen containing class ID (unique)
            region: ee.Geometry - Optional region to limit analysis
            class_name_property: str - Property/column containing class name
        """
        self.training_data = training_data
        self.image = image
        self.class_property = class_property
        self.class_name_property = class_name_property
        self.region = region
        self.band_names = self.image.bandNames().getInfo()
        self.class_mapping = None
    def get_display_property(self):
        """
        Helper functions to determine which properties (column in the training data) to display
        """
        return self.class_name_property if self.class_name_property else self.class_property
    def class_renaming(self):
        """
        Function to mapped between class ID and class names
        """
        if self.class_mapping is None and self.class_name_property:
            try:
                #mapped the combination between ID and names
                pairs = self.training_data.distinct([self.class_property, self.class_name_property])
                pair_info = pairs.getInfo()
                mapping = {}
                for features in pair_info['features']:
                    prop = features['properties']
                    class_id = prop[self.class_property]
                    class_name = prop[self.class_name_property]
                    # Convert float to int for consistent mapping keys
                    if isinstance(class_id, (int, float)):
                        mapping[int(class_id)] = class_name
                    else:
                        mapping[class_id] = class_name
                return mapping
            except Exception as e:
                print(f"Warning: Could not create class mapping: {e}")
                return {}
        return self.class_mapping or {}
    def add_class_names(self, df):
        """
        Add class names to dataframe based on class ids
        """ 
        id_column = self.class_property
        class_name = self.class_name_property  # Add this line
        mapping = self.class_renaming()
        if mapping:
            df[class_name] = df[id_column].astype(float).astype(int).map(lambda x: mapping.get(x, f"Class {x}"))
            cols = df.columns.tolist()
            if class_name in cols and id_column in cols:  
                id_idx = cols.index(id_column)
                cols.remove(class_name)  
                cols.insert(id_idx + 1, class_name) 
                df = df[cols]
        return df

    #Basic statistic of the training data:
    #Note that this process applied to training data before pixel value is extracted
    def sample_stats(self):
        """
        Get basic statistics about the training dataset. 
        """
        try:
            # Perfom in server side to minimize computational load
            class_counts = self.training_data.aggregate_histogram(self.class_property)
            total_samples = self.training_data.size()
            unique_classes = self.training_data.aggregate_array(self.class_property).distinct()
            # Get class names if available
            class_names = None
            if self.class_name_property:
                class_names = self.training_data.aggregate_histogram(self.class_name_property)            
            #return the dictionary
            results = ee.Dictionary({
                'class_counts': class_counts,
                'total_samples': total_samples,
                'unique_classes': unique_classes
            }).getInfo()
            if class_names:
                results['class_names'] = class_names            
            # Process results client-side
            class_counts_dict = results['class_counts']
            total_count = results['total_samples']
            classes_list = results['unique_classes']
            stats_dict = {
                'total_samples': total_count,
                'num_classes': len(classes_list),
                'class_counts': class_counts_dict,
                'classes': classes_list,
                'class_balance': {str(k): v/total_count for k, v in class_counts_dict.items()}
            }
            # Add class names mapping if available
            if 'class_names' in results:
                stats_dict['class_names'] = results['class_names']            
            return stats_dict
            
        except Exception as e:
            print(f"Error in get_basic_statistics: {e}")
            return None


# Step: 3.4 Preview of Verified Reference Data
# Sub-step: 3.4.1 Preview of Verified Reference Data
    def get_sample_stats_df(self):
        """
        Get sample statistics as a formatted DataFrame
        """
        stats = self.sample_stats()
        if not stats:
            return pd.DataFrame()
        
        # Create DataFrame from class counts
        stats_data = []
        for class_id, count in stats['class_counts'].items():
            proportion = stats['class_balance'][str(class_id)]
            stats_data.append({
                'Class_ID': class_id,
                'Sample_Count': count,
                'Proportion': proportion,
                'Percentage': proportion * 100
            })
        
        df = pd.DataFrame(stats_data)
        df = df.sort_values('Class_ID').reset_index(drop=True)
        df = df.rename(columns={'Class_ID': self.class_property})
        # Add class names if available
        df = self.add_class_names(df)
        
        # Format percentage column
        df['Percentage'] = df['Percentage'].round(2)
        df['Proportion'] = df['Proportion'].round(4)
        
        return df
