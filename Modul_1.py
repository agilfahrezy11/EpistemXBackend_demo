import ee
from datetime import datetime
import logging
ee.Initialize()
#Configure root for global functions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ReflectanceData:
    """
    Class for fetching and analyzing Landsat optical/thermal image collections with logging.
    """
    #Define the optical datasets. The band reflectances used is from Collection 2 Surface Reflectancce Data
    OPTICAL_DATASETS = {
        'L5_SR': {
            'collection': 'LANDSAT/LT05/C02/T1_L2',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_sr',
            'sensor': 'L5',
            'description': 'Landsat 5 Surface Reflectance',
        },
        'L7_SR': {
            'collection': 'LANDSAT/LE07/C02/T1_L2', 
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_sr',
            'sensor': 'L7',
            'description': 'Landsat 7 Surface Reflectance'
        },
        'L8_SR': {
            'collection': 'LANDSAT/LC08/C02/T1_L2',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_sr',
            'sensor': 'L8',
            'description': 'Landsat 8 Surface Reflectance'
        },
        'L9_SR': {
            'collection': 'LANDSAT/LC09/C02/T1_L2',
            'cloud_property': 'CLOUD_COVER_LAND', 
            'type': 'landsat_sr',
            'sensor': 'L9',
            'description': 'Landsat 9 Surface Reflectance'
        }
    }
#Define the thermal datasets. The thermal bands used is from Collection 2 Top-of-atmosphere data 
#The TOA data provide consistent result and contain minimum mising pixel data
    THERMAL_DATASETS = {
        'L5_TOA': {
            'collection': 'LANDSAT/LT05/C02/T1_TOA',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_toa',
            'sensor': 'L5',
            'description': 'Landsat 5 Top-of-atmosphere reflectance',
        },
        'L7_TOA': {
            'collection': 'LANDSAT/LE07/C02/T1_TOA',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_toa',
            'sensor': 'L7',
            'description': 'Landsat 7 Top-of-atmosphere reflectance',
        },
        'L8_TOA': {
            'collection': 'LANDSAT/LC08/C02/T1_TOA',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_toa',
            'sensor': 'L8',
            'description': 'Landsat 8 Top-of-atmosphere reflectance'  
        },
        'L9_TOA': {
            'collection': 'LANDSAT/LC09/C02/T1_TOA',
            'cloud_property': 'CLOUD_COVER_LAND',
            'type': 'landsat_toa',
            'sensor': 'L9',
            'description': 'Landsat 9 Top-of-atmosphere reflectance'      
        }
    }
    #Initialize the class
    def __init__(self, log_level=logging.INFO):
        """
        Initialize the ReflectanceData object and set up a class-specific logger.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        self.logger.info("ReflectanceData initialized.")
    #================== 1. Functions to compute image collection statistic ==================
    def get_collection_statistics(self, collection, compute_stats=True):
        """
        Get comprehensive statistics about an image collection.
        """
        #Get the number of image used 
        try:
            size = collection.size()
            if compute_stats:
                total_images = size.getInfo() #Client side operation, produce number of image collection
                if total_images > 0:
                    #Get the cloud cover percentage, and image aqcusition date
                    cloud_values = collection.aggregate_array('CLOUD_COVER_LAND').getInfo()
                    dates = collection.aggregate_array('system:time_start').getInfo()
                    dates_readable = [datetime.fromtimestamp(d/1000).strftime('%Y-%m-%d') for d in dates]
                    date_range = f"{min(dates_readable)} to {max(dates_readable)}"
                    #Get information regarding image's WRS path and row
                    try:
                        first_img = collection.first()
                        has_path_row = first_img.propertyNames().contains('WRS_PATH').getInfo()
                        if has_path_row:
                            paths = collection.aggregate_array('WRS_PATH').getInfo()
                            rows = collection.aggregate_array('WRS_ROW').getInfo()
                            path_rows = list(set(zip(paths, rows)))
                            path_rows.sort()
                        else:
                            path_rows = []
                    except Exception:
                        path_rows = []
                    #Image collections information 
                    stats = {
                        'total_images': total_images,
                        'date_range': date_range,
                        'cloud_cover': {
                            'min': min(cloud_values) if cloud_values else None,
                            'max': max(cloud_values) if cloud_values else None,
                            'mean': sum(cloud_values) / len(cloud_values) if cloud_values else None,
                            'values': cloud_values
                        },
                        'path_row_tiles': path_rows,
                        'unique_tiles': len(path_rows),
                        'individual_dates': dates_readable
                    }
                else:
                    stats = {
                        'total_images': 0,
                        'date_range': "No images found",
                        'cloud_cover': {'min': None, 'max': None, 'mean': None, 'values': []},
                        'path_row_tiles': [],
                        'unique_tiles': 0,
                        'individual_dates': []
                    }
            else:
                stats = {
                    'total_images': 'computed_on_demand',
                    'size_object': size,
                    'collection_object': collection,
                    'computed': False
                }
            return stats

        except Exception as e:
            self.logger.error(f"Error getting collection statistics: {str(e)}")
            return {'error': str(e)}
    
    #================== 2. Functions to fetch and perform filtering on image collection ==================
    def get_optical_data(self, aoi, start_date, end_date, optical_data='L8_SR',
                        cloud_cover=30,
                        verbose=True, compute_detailed_stats=True):
        """
        Get optical image collection for Landsat 5-9 SR data with detailed information logging.
        Parameters
        ----------
        aoi :  ee.FeatureCollection
            Area of interest.
        start_date : str
            Start date in format 'YYYY-MM-DD'.
        end_date : str
            End date in format 'YYYY-MM-DD'.
        optical_data : str
            Dataset type: 'L5_SR', 'L7_SR', 'L8_SR', 'L9_SR'.
        cloud_cover : int
            Maximum cloud cover percentage (default: 30).
        verbose : bool
            Print detailed information about the collection (default: True).
        compute_detailed_stats : bool
            If True, compute detailed statistics 
            If False, return only basic inf

        Returns
        -------
        tuple : (ee.ImageCollection, dict)
            Filtered and preprocessed image collection with statistics.
        """
        #Import masking and renaming functions for landsat surface reflectance data
        from optical_preprocessing import mask_landsat_sr, rename_landsat_bands

        if optical_data not in self.OPTICAL_DATASETS:
            raise ValueError(f"optical_data must be one of: {list(self.OPTICAL_DATASETS.keys())}")

        config = self.OPTICAL_DATASETS[optical_data]

        #Use verbose to import detailed logging information
        if verbose:
            self.logger.info(f"Starting data fetch for {config['description']}")
            self.logger.info(f"Date range: {start_date} to {end_date}")
            self.logger.info(f"Cloud cover threshold: {cloud_cover}%")
            if not compute_detailed_stats:
                self.logger.info("Fast mode enabled - detailed statistics will not be computed")

        #Initial collection
        initial_collection = (ee.ImageCollection(config['collection'])
                            .filterBounds(aoi)
                            .filterDate(start_date, end_date))

        initial_stats = self.get_collection_statistics(initial_collection, compute_detailed_stats)

        if verbose and compute_detailed_stats and initial_stats.get('total_images', 0) > 0:
            self.logger.info(f"Initial collection (before cloud filtering): {initial_stats['total_images']} images")
            self.logger.info(f"Date range of available images: {initial_stats['date_range']}")

        #Collection after cloud cover filter
        collection = initial_collection.filter(ee.Filter.lt(config['cloud_property'], cloud_cover))
        filtered_stats = self.get_collection_statistics(collection, compute_detailed_stats)
        #Computing image statistics
        if verbose and compute_detailed_stats:
            if filtered_stats.get('total_images', 0) > 0:
                self.logger.info(f"After cloud filtering (<{cloud_cover}%): {filtered_stats['total_images']} images")
                self.logger.info(f"Cloud cover of selected images: "
                                f"{filtered_stats['cloud_cover']['min']:.1f}% - "
                                f"{filtered_stats['cloud_cover']['max']:.1f}%")
                self.logger.info(f"Average cloud cover: {filtered_stats['cloud_cover']['mean']:.1f}%")
                if len(filtered_stats['individual_dates']) <= 20:
                    self.logger.info(f"Image dates: {', '.join(filtered_stats['individual_dates'])}")
                else:
                    self.logger.info(f"Images span from {min(filtered_stats['individual_dates'])} "
                                    f"to {max(filtered_stats['individual_dates'])}")

                if filtered_stats['path_row_tiles']:
                    path_row_str = ', '.join([f"{p}/{r}" for p, r in filtered_stats['path_row_tiles'][:10]])
                    if len(filtered_stats['path_row_tiles']) > 10:
                        path_row_str += f" ... (+{len(filtered_stats['path_row_tiles']) - 10} more)"
                    self.logger.info(f"Path/Row tiles: {path_row_str}")
            else:
                self.logger.warning(f"No images found matching criteria (cloud cover < {cloud_cover}%)")
                if initial_stats.get('total_images', 0) > 0:
                    self.logger.info(f"Consider increasing cloud cover threshold. "
                                    f"Available range: {initial_stats['cloud_cover']['min']:.1f}% - "
                                    f"{initial_stats['cloud_cover']['max']:.1f}%")
        elif verbose:
            self.logger.info("Filtered collection created (use compute_detailed_stats=True for more information)")

        #Apply masking and band renaming to image collection after filtering
        collection = (collection
                    .map(mask_landsat_sr)
                    .map(lambda img: rename_landsat_bands(img, config['sensor'], is_sr=True)))

        #Return results
        return collection, {
            'dataset': config['description'],
            'sensor': config['sensor'],
            'date_range_requested': f"{start_date} to {end_date}",
            'cloud_cover_threshold': cloud_cover,
            'initial_collection': initial_stats,
            'filtered_collection': filtered_stats,
            'detailed_stats_computed': compute_detailed_stats
        }


    def print_collection_summary(self, stats):
        """
        Print a formatted summary of collection statistics.
        """
        print("\n" + "="*60)
        print(f"LANDSAT DATA COLLECTION SUMMARY")
        print("="*60)
        print(f"Dataset: {stats['dataset']} ({stats['sensor']})")
        print(f"Date Range: {stats['date_range_requested']}")
        print(f"Cloud Cover Threshold: {stats['cloud_cover_threshold']}%")
        print("="*60 + "\n")

#================== 3. Functions to fetch and perform filtering on thermal bands ==================
    def get_thermal_bands(self, aoi, start_date, end_date, thermal_data = 'L8_TOA', cloud_cover=30,
                        verbose=True, compute_detailed_stats=True):
        """
        Get the thermal bands from landsat TOA data
        """
        from optical_preprocessing import mask_landsat_toa

        if thermal_data not in self.THERMAL_DATASETS:
            raise ValueError(f"thermal_data must be one of: {list(self.THERMAL_DATASETS.keys())}")

        config = self.THERMAL_DATASETS[thermal_data]

        #Decide which thermal band to select
        sensor = config['sensor']
        if sensor == 'L5':
            thermal_band = 'B6'
        elif sensor == 'L7':
            thermal_band = 'B6_VCID_2'
        elif sensor in ['L8', 'L9']:
            thermal_band = 'B10'
        else:
            raise ValueError(f"Unsupported sensor type: {sensor}")

        #Logging
        if verbose:
            self.logger.info(f"Starting thermal data fetch for {config['description']}")
            self.logger.info(f"Date range: {start_date} to {end_date}")
            self.logger.info(f"Cloud cover threshold: {cloud_cover}%")
            if not compute_detailed_stats:
                self.logger.info("Fast mode enabled - detailed statistics will not be computed")

        #Initial collection
        initial_collection = (ee.ImageCollection(config['collection'])
                            .filterBounds(aoi)
                            .filterDate(start_date, end_date))
        initial_stats = self.get_collection_statistics(initial_collection, compute_detailed_stats)

        if verbose and compute_detailed_stats and initial_stats.get('total_images', 0) > 0:
            self.logger.info(f"Initial collection (before cloud filtering): {initial_stats['total_images']} images")
            self.logger.info(f"Date range of available images: {initial_stats['date_range']}")

        #Apply cloud cover filter
        collection = initial_collection.filter(ee.Filter.lt(config['cloud_property'], cloud_cover))

        filtered_stats = self.get_collection_statistics(collection, compute_detailed_stats)

        if verbose and compute_detailed_stats:
            if filtered_stats.get('total_images', 0) > 0:
                self.logger.info(f"After cloud filtering (<{cloud_cover}%): {filtered_stats['total_images']} images")
                self.logger.info(f"Cloud cover range: {filtered_stats['cloud_cover']['min']:.1f}% - {filtered_stats['cloud_cover']['max']:.1f}%")
                self.logger.info(f"Average cloud cover: {filtered_stats['cloud_cover']['mean']:.1f}%")
            else:
                self.logger.warning(f"No images found matching criteria (cloud cover < {cloud_cover}%)")
        elif verbose:
            self.logger.info("Filtered collection created (use compute_detailed_stats=True for detailed info)")

        #Apply masking (QA-based)

        collection = collection.map(mask_landsat_toa)
        collection = collection.select(thermal_band)

        #Return collection and stats
        return collection, {
            'dataset': config['description'],
            'sensor': sensor,
            'thermal_band': thermal_band,
            'date_range_requested': f"{start_date} to {end_date}",
            'cloud_cover_threshold': cloud_cover,
            'initial_collection': initial_stats,
            'filtered_collection': filtered_stats,
            'detailed_stats_computed': compute_detailed_stats
        }
#================== 3. Temporal compositing ==================              
    def temporal_compositing(self, aoi, collection, data_type='ms'):
        """
        Perform spatio-temporal filtering of the image collection using reducers in earth engine
        Parameters:
            AOI: ee.FeatureCollection of the area of interest
            collection: ee.ImageCollection (multispectral bands or indices)
            data_type: str, 'MS' or 'idx'. (MS = multispectral bands, idx = spectral transformation bands)
        return:
        stacked ee.Image containing spatiotemporal bands 
        """
        data_type = data_type.lower()
        details = {}
        try:
            if data_type == 'ms':
                first_image = ee.Image(collection.first())
                available_bands = first_image.bandNames().getInfo()
                self.logger.info(f"Available bands: {available_bands}")
                required_bands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
                missing = [b for b in required_bands if b not in available_bands]
                if missing:
                    raise ValueError(f"Collection missing bands: {missing}")
                #first split the bands 
                blue_band = collection.select(['BLUE'])
                other_band = collection.select(['GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'])
                #second, define the reducer (can be customized accordingly. For this example, we will used mean, median, and variance)
                #restore+ original code use median, percentile, and standard deviation
                blue_band_reducer = ee.Reducer.median()
                other_band_reducer = (ee.Reducer.median()
                                    .combine(ee.Reducer.variance(), sharedInputs=True)
                                    .combine(ee.Reducer.mean(), sharedInputs=True))
                #count_reducer = ee.Reducer.count()
                #third, Apply the reducer
                blue_comp = blue_band.reduce(blue_band_reducer)
                other_comp = other_band.reduce(other_band_reducer)
                #valid_pixel_count = collection.select(['RED']).reduce(count_reducer)
                #Print the result
                #comp_stack = ee.Image.cat([blue_comp, other_comp, valid_pixel_count]).clip(aoi)
                comp_stack = ee.Image.cat([blue_comp, other_comp]).clip(aoi)
                band_count = comp_stack.bandNames().size().getInfo()
                self.logger.info(f"Final multispectral composite created with {band_count} bands")
                details = {
                    'Blue band filtered': blue_comp,
                    'Other bands filtered': other_comp,
                    #'Valid pixel count': valid_pixel_count,
                    'Band count': band_count
                }
                return comp_stack, details
            #if the temporal compositing is used for spectral indices
            elif data_type == 'idx':
                indices_reducer = ee.Reducer.median()
                indices_comp = collection.reduce(indices_reducer)
                clip_indices_comp = indices_comp.clip(aoi)
                band_count = clip_indices_comp.bandNames().size().getInfo()
                self.logger.info(f"Final indices composite created with {band_count} bands")
                details = {
                    'Indices composite': clip_indices_comp,
                    'Band count': band_count
                }
                return clip_indices_comp, details
            else:
                raise ValueError('Unsupported data_type. Use "ms" for multispectral bands and "idx" for spectral transformation indices')
        except Exception as e:
            self.logger.error(f"Error in temporal_compositing: {str(e)}")
            raise
