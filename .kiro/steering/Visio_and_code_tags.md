<!------------------------------------------------------------------------------------
    This document serves as a tags for connecting MS Visio Module Workflow with python codes 
    in this project
-------------------------------------------------------------------------------------> 
# Helper Functions to support each module
   ## Shapefile validation and conversion
      ├── src/                      
      │   └── module_helpers.py
      │         ├── def init_gee
      │         ├── class shapefile_validator
      │         │     ├── def validate_and_fix_geometry
      │         │     ├── def _fix_crs
      │         │     ├── def _clean_geometries
      │         │     ├── def _validate_points
      │         │     ├── def _validate_polygons
      │         │     ├── def _is_valid_coordinate
      │         │     ├── def _count_vertices
      │         │     └── def _final_validation
      │         ├── class EE_converter
      │         │      ├── def __init__
      │         │      ├── def log  
      │         │      ├── def convert_aoi_gdf        
      │         │      └── def convert_roi_gdf        



# Module 1: Acquisition of Near-Cloud-Free Satellite Imagery
   ## System Response 1.1: Area of Interest Definition
   ```
      ├── src/                      
      │   ├── module_helpers.py
      │         ├── class shapefile_validator
      │         └── class EE_converter
   ```
   ## System Response 1.2: Search and Filter Imagery
   ```
      ├── src/                       
      │   ├── src_modul_1.py
      │         ├── class Reflectance_Data
      │         └── class Reflectance_Stats  
   ```
   ## System Response 1.3: Imagery Download
   ```
      ├── pages/                     
      │   ├── 1_Module_1_Generate_Image_Mosaic.py
      │            └── line 389 - 443 
      │            (if st.button("Start Export to Google Drive", type="primary"))
   ```

# Module 2: Determining LULC Classification Schema and Classes
   ## System Response 2.1a: Upload Classification Scheme
      ├── src/                      
      │   └── src_modul_2.py
      │         └── Class LULCSchemeClass
      │               ├──  def process_csv_upload
      │               ├──  def finalize_csv_upload
      │               └──  def auto_detect_csv_columns
   
   ## System Response 2.1b: Manual Scheme Definition
      ├── src/                      
      │   └── src_modul_2.py
      │         └── Class LULC_Scheme_Manager:
      │               ├──  def validate_class_input
      │               ├──  def add_class
      │               ├──  def _reset_edit_mode
      │               ├──  def _sort_and_update_next_id
      │               ├──  def edit_class
      │               ├──  def delete_class
      │               └──  def cancel_edit
      ├── pages/                     
      │         └── 2_Module_2_Classification_scheme.py
      │               └── def render_manual_input_form

   ## System Response 2.1c: Template Classification Scheme
      ├── src/                      
      │   ├── src_modul_2.py
      │   │       ├── Class LULC_Scheme_Manager:
      │   │       │     ├──  def load_default_scheme
      │   │       │     └──  def get_default_scheme

   ## System Response 2.1c: Download classification scheme
      ├── src/                      
      │   ├── src_modul_2.py
      │   │      ├── Class LULC_Scheme_Manager:
      │   │      │     └──  def get_csv_data
      │   ├── 2_Module_2_Classification_scheme.py
      │   │      └── st.download_button (line 265 - 285)

# Module 3: Training Data Generation


# Module 4: Spectral Separability Analysis

   ## System Response 4.1 Separability Analysis
      ├── src/                      
      │   └── src_modul_4.py
      │                └── Class sample_quality
      │                     ├──  def get_display_property
      │                     ├──  def class_renaming
      │                     ├──  def add_class_names
      │                     ├──  def sample_stats
      │                     ├──  def get_sample_stats_df
      │                     ├──  def extract_spectral_values
      │                     ├──  def sample_pixel_stats
      │                     ├──  def get_sample_pixel_stats_df
      │                     └──  def check_class_separability
      │                           ├──  def _jeffries_matusita_distance
      │                           └──  def transform_divergence                     
      │                     ├──  def get_separability_df
      │                     ├──  def lowest_separability
      │                     ├──  def separability_level
      │                     ├──  def sum_separability
      │                     └──  def print_analysis_summary

   ## System Response 4.2 Sample Visualization
      ├── src/                      
      │   └── src_modul_4_part2.py
      │                └── Class spectral_plotter
      │                     ├──  def plot_histogram
      │                     ├──  def plot_boxplot
      │                     ├──  def interactive_scatter_plot
      │                     ├──  def static_scatter_plot  
      │                     │      └──  def add_elipse
      │                     └──  def scatter_plot_3d                       

# Module 5: Improving Model Quality with Multi-Source Data

# Module 6: Land Use Land Cover Map Generation
   ## System Response 6.1 Prerequisites Check
      ├── pages/                     
      │   └── 4_Module_6_Classification and LULC Creation.py
      │         ├── with col1: (composite check: line 38 - 54)
      │         └── with col2: (training data check: line 57 - 81)


   ## System Response 6.2 Classification
      ├── src/                      
      │   └── src_modul_6.py
      │          ├── Class FeatureExtraction
      │          │    ├──  def stratified_split
      │          │    └──  def random_split                            
      │          ├── Class Generate_LULC:
      │          │    ├── def hard_classification
      │          │    └── def soft_classification

   ## System Response 6.3 Model Evaluation
      ├── src/                      
      │   └── src_modul_6.py
      │          ├── Class Generate_LULC:
      │          │    ├── def get_feature_importance
      │          │    └── def evaluate_model    

# Module 7: Thematic Accuracy Assessment
   ## System Response 7.1 Prerequisite check
      ├── pages/                     
      │   └── 5_Module_7_Thematic_Accuracy.py
      │         └── def check_prerequisites   
      ├── src/ 
          └── src_modul_7.py
            └──  def validate_inputs

   ## System Response 7.2 Ground Reference Verification
      ├── pages/                     
      │   └── 5_Module_7_Thematic_Accuracy.py
      │         └── def process_shapefile_upload

   ## System Response 7.3 Thematic Accuracy Assessment
      ├── src/                      
      │   └── src_modul_7.py
      │          └── Class Thematic_Accuracy_Assessment
      │               ├── def _calculate_accuracy_confidence_interval
      │               ├── def _calculate_f1_scores
      │               ├── def _extract_confusion_matrix_data
      │               ├── def run_accuracy_assessment                
      │               └── def format_accuracy_summary

# Module 8: House