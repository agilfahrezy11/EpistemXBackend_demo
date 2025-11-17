<!------------------------------------------------------------------------------------
    This document serves as a tags for connecting Quarto Technical Documentation with python codes 
    in this project
-------------------------------------------------------------------------------------> 

EpistemXBackend
├── docs
	├── repo_structure.md	       # Repository navigation structure
├── src/epistemx				  # contain functions of backend
	├── __init__.py                      
	├── helpers.py                # helper functions shared by multiple modules
	├── shapefile_utils.py        # to validate the uploaded shapefiles
	├── modules_phase1
		└── module_1.py           # Acquisition of Near-Cloud-Free Satellite Imagery
		└── module_3.py           # Reference Data Generation
		└── module_4.py           # Spectral Separability Analysis
		└── module_6.py           # LULC Map Generation
		└── module_7.py           # Thematic Accuracy Assessment
├── test_data
	├── AOI_sample.zip         		# Test shapefiles/CSV for AOI/training (small files)
├── notebooks
	└── Module_implmentation.ipynb  # Exploratory analysis for developers