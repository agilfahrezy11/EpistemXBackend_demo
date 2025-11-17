# module6.py

import ee
import pandas as pd
from tqdm import tqdm

ee.Initialize()

# Module 6: LULC Map Generation
# Module Overview
# This module provides functionality for users to generate the final LULC map as the primary output product, 
# derived from the validated and optimized classification model developed in Modules 1-5. 
# Before the processing begins, it ensures that all required inputs - such as satellite imagery, 
# classification schemes, and training samples - are properly prepared and validated. 
# Users can specify data splits, select classification modes, perform hyperparameter tuning optimization, 
# and apply the trained model to the entire AOI to produce a high-quality, georeferenced LULC map. 
# The module supports Random Forest (RF) based classification for adaptability, 
# aligning with hierarchical or non-hierarchical schemes from Module 2, 
# and incorporates feedback from optional covariate enhancements in Module 5. 
# Upon completion, users receive a displayable, analyzable, and downloadable map with metadata, 
# enabling applications such as deforestation monitoring or conservation planning.

# Input
# User’s Input
# 1. RF hyperparameters: number of trees (n_tree), variables per split (var_split), minimum leaf population (min_leaf_pop).
# 2. Choice of hyperparameter tuning option (e.g., Hard Classification Tuning, K-Fold Soft Classification Tuning, etc.).
# Automatic System Input
# 1. None.
# Input from Other Modules
# 1. Near-cloud-free satellite imagery within the AOI (Module 1).
# 2. LULC classification scheme table (Module 2).
# 3. Georeferenced reference data (training + validation samples)(Module 3).
# 4. Validated sample data and separability results (Module 4).

# Output
# * Trained RF classification model.
# * Classified LULC raster map.
# * Optimal hyperparameters from tuning process
# * Model accuracy report if validation data available (including overall accuracy, precision, recall, F1-score, Kappa, model error matrix).

# Process
# Checking Prerequisites from Previous Modules
# Front-end: The system verifies all required inputs are available and prompts users to complete previous modules if reference data or other essential inputs are missing.
# Back-end:
# 1. Validates availability of satellite imagery, classification scheme, and reference data
# 2. Checks reference data partitioning from Module 3
# 3. Ensures compatibility between all input datasets
# 4. If validation data is unavailable, guides user to Module 3 for completion

# Define Classification Specifications
# Front-end: User configures Random Forest parameters:
# * Number of trees (n_tree): Controls ensemble size (default: 100). More trees increase stability but require more computation.
# * Variables per split (var_split): Features considered at each decision point. Default: square root of total features.
# * Minimum leaf population (min_leaf_pop): Minimum samples in terminal nodes (default: 1). Higher values prevent overfitting.
# Back-end:
# 1. Stores user-specified hyperparameters
# 2. Prepares for hyperparameter tuning process
# 3. Integrates data splitting strategies from FeatureExtraction class

class FeatureExtraction:
    """
    Perform feature extraction as one of the input for land cover classification. Three types of split is presented here:
    Random Split: splitting the input data randomly based on specified split ratio
    stratified random split: splitting the input data randomly based on strata (lulc class). More representative for many cases
    statified k_fold split: splitting the input data into folds 
    """
    def __init__(self):
        """
        Initializing the class function
        """
############################# 1. Single Random Split ###########################
#extract pixel value for the labeled region of interest and partitioned them into training and testing data
#This can be used if the training/reference data is balanced across class and required more fast result
    def random_split(self, image, roi, class_property, split_ratio = 0.6, pixel_size = 10, tile_scale=16):
        """
        Perform single random split and extract pixel value from the imagery
            Parameters:
                image = ee.Image
                aoi = area of interest, ee.FeatureCollection
                split_ratio = 
            Returns:
                tuple: (training_samples, testing_samples)
        """
        #create a random column
        roi_random = roi.randomColumn()
        #partioned the original training data
        training = roi_random.filter(ee.Filter.lt('random', split_ratio))
        testing = roi_random.filter(ee.Filter.gte('random', split_ratio))
        #extract the pixel values
        training_pixels = image.sampleRegions(
                            collection=training,
                            properties = [class_property],
                            scale = pixel_size,
                            tileScale = tile_scale 
        )
        testing_pixels = image.sampleRegions(
                            collection=testing,
                            properties = [class_property],
                            scale = pixel_size,
                            tileScale = tile_scale 
        )
        print('Single Random Split Training Pixel Size:', training_pixels.size().getInfo())
        print('Single Random Split Testing Pixel Size:', testing_pixels.size().getInfo())
        return training_pixels, testing_pixels
    ############################## 2. Strafied Random Split ###########################
    # Conduct stratified train and test split, ideal for proportional split of the data
    def stratified_split (self, roi, image, class_prop, pixel_size= 10, train_ratio = 0.7, seed=0):
        """
        Used stratified random split to partitioned the original sample data into training and testing data used for model development
        Args:
            Split the region of interest using a stratified random approach, which use class label as basis for splitting
            roi: ee.FeatureCollection (original region of interest)
            class_prop: Class property (column) contain unique class ID
            tran_ratio: ratio for train-test split (usually 70% for training and 50% for testing)
        Return:
        ee.FeatureCollection, consist of training and testing data
        
        """
        #Define the unique class id using aggregate array
        classes = roi.aggregate_array(class_prop).distinct()
        #split the region of interest based on the class
        def split_class (c):
            subset = (roi.filter(ee.Filter.eq(class_prop, c))
                    .randomColumn('random', seed=seed))
            train = (subset.filter(ee.Filter.lt('random', train_ratio))
                        .map(lambda f: f.set('fraction', 'training')))
            test = (subset.filter(ee.Filter.gte('random', train_ratio))
                        .map(lambda f: f.set('fraction', 'testing')))
            return train.merge(test)
        #map the function for all the class
        split_fc = ee.FeatureCollection(classes.map(split_class)).flatten()
        #filter for training and testing
        train_fc = split_fc.filter(ee.Filter.eq('fraction', 'training'))
        test_fc = split_fc.filter(ee.Filter.eq('fraction', 'testing'))
        print('Stratified Random Split Training Pixel Size:', train_fc.size().getInfo())
        print('Stratified Random Split Testing Pixel Size:', test_fc.size().getInfo())      
        #sample the image based stratified split data
        train_pix = image.sampleRegions(
                            collection=train_fc,
                            properties = [class_prop],
                            scale = pixel_size,
                            tileScale = 16)
        test_pix = image.sampleRegions(
                            collection = test_fc,
                            properties = [class_prop],
                            scale = pixel_size,
                            tileScale = 16
        )
  
        return train_pix, test_pix
############################# 3. Stratified K-fold Split ###########################
# the strafied kfold cross validation split for more robust partitioning between training and validation data.
# Ideal for imbalance dataset. 
    def stratified_kfold(self, samples, class_property, k=5, seed=0):
        """
        Perform stratified kfold cross-validation split on input reference data.
        
        Parameters:
            samples (ee.FeatureCollection): training data or reference data which contain unique class label ID
            class_property (str): column name contain unique label ID
            k (int): Number of folds.
            seed (int): Random seed for reproducibility.
        
        Returns:
            ee.FeatureCollection: A collection of k folds. Each fold is a Feature
                                with 'training' and 'validation' FeatureCollections.
        """
        #define the logic for k-fold. It tells us how wide the split will be
        step = 1.0 / k
        #Threshold are similar to split ratio, in this context, an evenly space of data numbers. The results is a cut points for the folds,
        #in which each fold will takes sample whose asigned random number within the ranges
        thresholds = ee.List.sequence(0, 1 - step, step)
        #This code will aggregate unique class label into one distinct label
        classes = samples.aggregate_array(class_property).distinct()
        #function for create the folds using the given threshold
        def make_fold(threshold):
            threshold = ee.Number(threshold)
            #Split each class into training and testing, based on random numbers
            #each class split ensure startification during split
            def per_class(c):
                c = ee.Number(c)
                subset = samples.filter(ee.Filter.eq(class_property, c)) \
                                .randomColumn('random', seed)
                training = subset.filter(
                    ee.Filter.Or(
                        ee.Filter.lt('random', threshold),
                        ee.Filter.gte('random', threshold.add(step))
                    )
                )
                testing = subset.filter(
                    ee.Filter.And(
                        ee.Filter.gte('random', threshold),
                        ee.Filter.lt('random', threshold.add(step))
                    )
                )
                return ee.Feature(None, {
                    'training': training,
                    'testing': testing
                })
            #Applied the splits for each class in the dataset
            splits = classes.map(per_class)
            # merge all classes back together for this fold
            # merged all classes in the training subset
            training = ee.FeatureCollection(splits.map(lambda f: ee.Feature(f).get('training'))).flatten()
            # merge all classes in the testing subset
            testing = ee.FeatureCollection(splits.map(lambda f: ee.Feature(f).get('testing'))).flatten()
            return ee.Feature(None, {'training': training, 'testing': testing})

        folds = thresholds.map(make_fold)
        # Print overall k-fold information (moved outside the mapped function)
        print(f'Created {k} folds for stratified k-fold cross-validation')
        print('Total input samples:', samples.size().getInfo())
        print(f'Each fold will have approximately {samples.size().divide(k).getInfo():.0f} samples for validation')        
        return ee.FeatureCollection(folds)
    ############################# 4. Inspecting the folds ###########################
    # Used to inspect the resulr of stratified k-fold size
    def inspect_fold(self, folds, fold_index):
        """Inspect a specific fold's sizes"""
        fold = ee.Feature(folds.toList(1, fold_index).get(0))
        train = ee.FeatureCollection(fold.get('training'))
        val = ee.FeatureCollection(fold.get('testing'))
        print(f'Fold {fold_index + 1} - Training: {train.size().getInfo()}, Testing: {val.size().getInfo()}')
        return train, val  

# Hyperparameter Tuning
# Front-end: User selects a tuning strategy. System evaluates multiple parameter combinations using training/validation split from Module 3. Results display optimal parameters and accuracy.
# Back-end:
# * Uses held-out validation data to assess performance.
# * If 100% of reference data allocated to training in Module 3 → no validation data available → tuning disabled.
# * User must upload independent validation data in Module 7.

class Hyperparameter_tuning:
    """
    Perform hyperparameter optimization for random forest classifiers. Several optimization are presented for different training data:
        1. Hard Classification tuning: This functions is used if the classification approach is hard multiclass classification
        2. Soft classification tuning: This functions is used if the classification approach is One-vs-Rest Binary classification framework
        3. Hard fold classification tuning: This function is used for multi-class classification with k-fold data
        4. Soft fold classification tuning: This functions is used if the classification approach is One-vs-Rest Binary classification framework with kfold data
    """
    def __init__(self):
        """
        Initialize the hyperparameter tuning class
        """
        pass
    ############################# 1. Multiclass Hard Classification Tuning ###########################
    # Hard Classification Tuning
    # Uses multiclass classification approach and evaluates performance using overall accuracy metric.
    # Related Function(s): Hard_classification_tuning() performs grid search for direct multiclass classification, testing all parameter combinations and selecting those with highest accuracy.
    def Hard_classification_tuning(self, train, test, image, class_property, 
                                   n_tree_list, var_split_list, min_leaf_pop_list):
        """
        Perform manual testing to find a set of parameters that yielded highest accuracy for Random Forest Classifier.
        Three main parameters were tested, namely Number of trees (n_tree), number of variable selected at split (var_split), and minimum sample population at leaf node (min_leaf_pop)
        This function is used for multiclass classification with training data from single or stratified random split
        Parameters:
            train: Training pixels
            test: Testing pixels
            image: ee.image for used for classification
            class_property: distinct labels in the training and testing data
            n_tree_list: list containing n_tree value for testing
            var_split_list: list containing var_split value for testing
            min_leaf_pop_list : list containing min_leaf_pop value for testing
        Returns:
        Best parameters combinations and resulting model accuracy (panda dataframe)
        """
        result = [] #initialize empty dictionary for storing parameters and accuracy score
        total_combinations = len(n_tree_list) * len(var_split_list) * len(min_leaf_pop_list)
        print(f"Starting hyperparameter tuning with {total_combinations} parameter combinations...")

        #manually test the classifiers, while looping through the parameters set
        with tqdm(total=total_combinations, desc="Hard Classification Tuning") as pbar:
            for n_tree in n_tree_list:
                for var_split in var_split_list:
                    for min_leaf_pop in min_leaf_pop_list:
                        try:
                            print(f"Testing: Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                            #initialize the random forest classifer
                            clf = ee.Classifier.smileRandomForest(
                                numberOfTrees=n_tree,
                                variablesPerSplit=var_split,
                                minLeafPopulation = min_leaf_pop,
                                seed=0
                            ).train(
                                features=train,
                                classProperty=class_property,
                                inputProperties=image.bandNames()
                            )
                            #Used partitioned test data, to evaluate the trained model
                            classified_test = test.classify(clf)
                            #test using error matrix
                            error_matrix = classified_test.errorMatrix(class_property, 'classification')
                            #append the result of the test
                            accuracy = error_matrix.accuracy().getInfo()
                            result.append({
                                'numberOfTrees': n_tree,
                                'variablesPerSplit': var_split,
                                'MinimumleafPopulation':min_leaf_pop,
                                'accuracy': accuracy
                            })
                            #print the message if error occur
                        except Exception as e:
                            print(f"Failed for Trees={n_tree}, Variable Split={var_split}, mininum leaf population = {min_leaf_pop}")
                            print(f"Error: {e}")
                            
                        finally:
                            pbar.update(1)
            #Convert the result into panda dataframe and print them
            if result:
                result_df = pd.DataFrame(result)
                result_df_sorted = result_df.sort_values(by='accuracy', ascending=False)#.reset_index(drop=True)
                
                print("\n" + "="*50)
                print("GRID SEARCH RESULTS")
                print("="*50)
                print("\nBest parameters (highest model accuracy):")
                print(result_df_sorted.iloc[0])
                print("\nTop 5 parameter combinations:")
                print(result_df_sorted.head())
                
                return result, result_df_sorted
            else:
                print("No successful parameter combinations found!")
                return [], pd.DataFrame()
        
    ############################# 2. Binary One-vs-rest soft Classification Tuning ###########################
    # Soft Classification Tuning
    # Uses one-vs-rest binary classification approach and evaluates performance using cross-entropy loss.
    # Related Function(s): Soft_classification_tuning() optimizes parameters for probabilistic classification, selecting combinations with lowest cross-entropy loss.
    def Soft_classification_tuning(self, train, test, image, class_property, 
                                   n_tree_list, var_split_list, min_leaf_pop_list, seed = 13):
        """
        Perform manual testing to find a set of parameters that yielded highest accuracy for Random Forest Classifier.
        Three main parameters were tested, namely Number of trees (n_tree), number of variable selected at split (var_split), and minimum sample population at leaf node (min_leaf_pop)
        This function is used for one-vs-rest binary classification with training data from single or stratified random split
        Parameters:
            train: Training pixels
            test: Testing pixels
            image: ee.image for used for classification
            class_property: distinct labels in the training and testing data
            n_tree_list: list containing n_tree value for testing
            var_split_list: list containing var_split value for testing
            min_leaf_pop_list : list containing min_leaf_pop value for testing
        Returns:
        Best parameters combinations and resulting model cross-entropy loss (panda dataframe)
        """
        #create an empty list to store all of the result
        result = []
        #get unique class ID
        class_list = train.aggregate_array(class_property).distinct()
        total_combinations = len(n_tree_list) * len(var_split_list) * len(min_leaf_pop_list)
        print(f"Starting soft classification tuning with {total_combinations} parameter combinations...")        
        #create a loop exploring all possible combination of parameter
        with tqdm(total=total_combinations, desc="Hard Classification Tuning") as pbar:  
            for n_tree in n_tree_list:
                for var_split in var_split_list:
                    for min_leaf_pop in min_leaf_pop_list: 
                        try:
                            print(f"Testing: Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                            def per_class(class_id):
                                class_id = ee.Number(class_id)

                                binary_train = train.map(lambda ft: ft.set(
                                    'binary', ee.Algorithms.If(
                                        ee.Number(ft.get(class_property)).eq(class_id), 1, 0
                                    )
                                ))
                                binary_test = test.map(lambda ft: ft.set(
                                    'binary', ee.Algorithms.If(
                                        ee.Number(ft.get(class_property)).eq(class_id), 1, 0
                                    )
                                ))
                                #Random Forest Model, set to probability mode
                                clf = (ee.Classifier.smileRandomForest(
                                        numberOfTrees = n_tree,
                                        variablesPerSplit = var_split,
                                        minLeafPopulation = min_leaf_pop,
                                        seed = seed)
                                        .setOutputMode('PROBABILITY'))
                                model = clf.train(
                                    features = binary_train,
                                    classProperty = 'binary',
                                    inputProperties = image.bandNames()
                                )
                                test_classified = binary_test.classify(model)

                                # Extract true class labels and predicted probabilities
                                y_true =  test_classified.aggregate_array('binary')
                                y_pred =  test_classified.aggregate_array('classification')
                                paired = y_true.zip(y_pred).map(
                                        lambda xy: ee.Dictionary({
                                            'y_true': ee.List(xy).get(0),
                                            'y_pred': ee.List(xy).get(1)
                                        })
                                    )
                                # function to calculate log loss(need clarification)
                                def log_loss (pair_dict):
                                    pair_dict = ee.Dictionary(pair_dict)
                                    y = ee.Number(pair_dict.get('y_true'))
                                    p = ee.Number(pair_dict.get('y_pred'))
                                    #epsilon for numerical stability
                                    epsilon = 1e-15
                                    p_clip = p.max(epsilon).min(ee.Number(1).subtract(epsilon))
                                    # Log loss formula: -[y*log(p) + (1-y)*log(1-p)]
                                    loss = y.multiply(p_clip.log()).add(
                                        ee.Number(1).subtract(y).multiply(
                                            ee.Number(1).subtract(p_clip).log()
                                        )
                                    ).multiply(-1)
                                    return loss

                                #Calculate log losses for all test samples
                                loss_list = paired.map(log_loss)
                                avg_loss = ee.Number(loss_list.reduce(ee.Reducer.mean()))
                                return avg_loss
                            #mapped the log loss for all class
                            loss_list = class_list.map(per_class)
                            avg_loss_all = ee.Number(ee.List(loss_list).reduce(ee.Reducer.mean()))
                            #get actuall loss value:
                            act_loss = avg_loss_all.getInfo()
                            #append the results of the tuning
                            result.append({
                                        'Number of Trees': n_tree,
                                        'Variable Per Split': var_split,
                                        'Minimum Leaf Populaton': min_leaf_pop,
                                        'Average Model Cross Entropy Loss': act_loss
                            })
                            print(f"Loss: {act_loss:.6f}")
                            pbar.update(1)

                            # Print this message if failed
                        except Exception as e:
                            print(f"Failed for Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                            print(f"Error: {e}")
                            pbar.update(1)
                            continue
            #convert the result into panda dataframe and viewed the best parameters
            if result:
                result_df = pd.DataFrame(result)
                result_df_sorted = result_df.sort_values(by='Average Model Cross Entropy Loss', ascending=True).reset_index(drop=True)
                
                print("\n" + "="*50)
                print("GRID SEARCH RESULTS")
                print("="*50)
                print("\nBest parameters (lowest loss):")
                print(result_df_sorted.iloc[0])
                print("\nTop 5 parameter combinations:")
                print(result_df_sorted.head())
                
                return result, result_df_sorted
            else:
                print("No successful parameter combinations found!")
                return [], pd.DataFrame
                #return [], pd.DataFrame()
    ############################# 3. Hard Multiclass Classification with k-fold data ###########################
    # K-Fold Cross Validation
    # Back-end: Performs robust parameter optimization using stratified k-fold cross-validation:
    # * Hard K-Fold: hard_tuning_kfold() for multiclass classification with k-fold validation
    # * Soft K-Fold: soft_tuning_kfold() for one-vs-rest classification with k-fold validation
    # This approach provides more reliable parameter estimates, especially for imbalanced datasets.
    def hard_tuning_kfold(self, reference_fold, image, class_prop,  
                         n_tree_list, var_split_list, min_leaf_pop_list, tile_scale=16, pixel_size = 10):
        """
        Perform manual testing to find a set of parameters that yielded highest accuracy for Random Forest Classifier with stratified k-fold input data
        This function is used if stratified k-fold split is used to partitioned the samples
        parameters: 
            reference_fold: ee.featurecollection result from stratified kfold
            image: ee.image remote sensing data
            class_prop: property name for class labels
            n_tree_list: list of int, number of trees to test
            v_split_list: list of int, number of variables to test
            leaf_pop_list: list of int, minimum leaf population to test
            tile_scale: scale parameter for sampling
        return: list of dict with parameters and average accuracy
        """ 
        #define and set the previous fold result
        k = reference_fold.size().getInfo()
        fold_list = reference_fold.toList(k)
        total_combinations = len(n_tree_list) * len(var_split_list) * len(min_leaf_pop_list)
        print(f"Starting soft classification tuning with {total_combinations} parameter combinations...")        
        result = []
        # Pre-sample regions for each fold (optimization)
        print("Pre-sampling regions for all folds...")
        fold_samples = []
        for i in range(k):
            fold = ee.Feature(fold_list.get(i))
            training_fc = ee.FeatureCollection(fold.get('training'))
            testing_fc = ee.FeatureCollection(fold.get('testing'))
            
            train_pixels = image.sampleRegions(
                collection=training_fc,
                properties=[class_prop],
                scale=pixel_size,
                tileScale=tile_scale
            )
            test_pixels = image.sampleRegions(
                collection=testing_fc,
                properties=[class_prop],
                scale=pixel_size,
                tileScale=tile_scale
            )
            fold_samples.append({'train': train_pixels, 'test': test_pixels})
        print("Sampling complete!")        
        #Create a gridsearch tuning by manually looped through the parameter space
        with tqdm (total=total_combinations, desc="Hard K-Fold Tuning") as pbar:
            for n_tree in n_tree_list:
                for var_split in var_split_list:
                    for min_leaf_pop in min_leaf_pop_list:
                        print(f"Testing: Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                        fold_acc_list = []
                        for i in range (k):
                            try: 
                                
                                    #classifier set to multiclass hard classification
                                clf = ee.Classifier.smileRandomForest(
                                    numberOfTrees=n_tree,
                                    variablesPerSplit=var_split,
                                    minLeafPopulation=min_leaf_pop,
                                    seed=0
                                    ).train(
                                    features=train_pixels,
                                    classProperty=class_prop,
                                    inputProperties=image.bandNames()
                                    )
                                    #function to evaluate the model
                                classified_val = test_pixels.classify(clf)
                                model_val = classified_val.errorMatrix(class_prop, 'classification')
                                fold_accuracy = model_val.accuracy().getInfo()
                                fold_acc_list.append(fold_accuracy)
                                

                            except Exception as e:
                                print(f"Failed for fold {i}")
                                print(f"Failed for Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                                print(f"Error: {e}")
                                continue
                        
                        # Calculate average accuracy across all folds for this parameter combination
                        if fold_acc_list:
                            avg_acc = sum(fold_acc_list) / len(fold_acc_list)
                            #Put the result into a list
                            result.append({
                                'Number of Trees': n_tree,
                                'Variable Per Split': var_split,
                                'Minimum Leaf Population': min_leaf_pop,
                                'Average Model Accuracy': avg_acc
                            })
                            print(f"Average Accuracy: {avg_acc:.6f}")
                        else:
                            print(f"[WARNING] No valid accuracy scores for parameter combination Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                        pbar.update(1)                            
            #convert to panda data frame and print the result
        if result:
                result_pd = pd.DataFrame(result)
                result_pd_sorted = result_pd.sort_values(by='Average Model Accuracy', ascending=False).reset_index(drop=True)
                print("\n" + "="*50)
                print("GRID SEARCH RESULTS")
                print("="*50)
                print("\nBest parameters (highest accuracy):")
                print(result_pd_sorted.iloc[0])
                print("\nTop 5 Parameter combinations:")
                print(result_pd_sorted.head())         
                return result, result_pd_sorted
        else:
                print("No successful parameter combinations found!")
                return [], pd.DataFrame()
        
    def soft_tuning_kfold(self, folds, image, class_property, 
                          n_tree_list, var_split_list, min_leaf_pop_list, seed=0, pixel_size = 10, tile_scale=16):
        """
        Perform manual testing to find a set of parameters that yielded lowest cross-entropy loss for Random Forest Classifier with k-fold data.
        This function is used for one-vs-rest binary classification with k-fold cross-validation
        
        Parameters:
            folds: ee.FeatureCollection result from stratified k-fold split
            image: ee.Image remote sensing data for classification
            class_property: property name for class labels
            n_tree_list: list of int, number of trees to test
            var_split_list: list of int, number of variables per split to test
            min_leaf_pop_list: list of int, minimum leaf population to test
            seed: random seed for reproducibility
            tile_scale: scale parameter for sampling
            
        Returns:
            tuple: (result list, sorted DataFrame with parameters and average cross-entropy loss)
        """    
        #define and set the previous fold result
        k = folds.size().getInfo()
        fold_list = folds.toList(k)
        
        #get the list of unique class id
        first_fold = ee.Feature(fold_list.get(0))
        training_fc0 = ee.FeatureCollection(first_fold.get('training'))
        classes = training_fc0.aggregate_array(class_property).distinct().getInfo()
        print(f"Classes found: {classes}")
        
        result = []
        
        # Calculate total combinations for progress tracking
        total_combinations = len(n_tree_list) * len(var_split_list) * len(min_leaf_pop_list)
        print(f"Starting soft k-fold tuning with {total_combinations} parameter combinations across {k} folds...")
        
        # Pre-sample regions for each fold and each class (optimization)
        print("Pre-sampling regions for all folds and classes...")
        fold_class_samples = []
        for i in range(k):
            fold = ee.Feature(fold_list.get(i))
            training_fc = ee.FeatureCollection(fold.get('training'))
            testing_fc = ee.FeatureCollection(fold.get('testing'))
            
            class_samples = {}
            for class_id in classes:
                # Create binary training and testing data for this class
                binary_train = training_fc.map(lambda ft: ft.set(
                    'binary', ee.Algorithms.If(
                        ee.Number(ft.get(class_property)).eq(class_id), 1, 0
                    )
                ))
                binary_test = testing_fc.map(lambda ft: ft.set(
                    'binary', ee.Algorithms.If(
                        ee.Number(ft.get(class_property)).eq(class_id), 1, 0
                    )
                ))
                
                # Sample the image for this class
                train_pixels = image.sampleRegions(
                    collection=binary_train,
                    properties=['binary'],
                    scale=pixel_size,
                    tileScale=tile_scale
                )
                test_pixels = image.sampleRegions(
                    collection=binary_test,
                    properties=['binary'],
                    scale=pixel_size,
                    tileScale=tile_scale
                )
                
                class_samples[class_id] = {'train': train_pixels, 'test': test_pixels}
            
            fold_class_samples.append(class_samples)
        print("Sampling complete!")
        
        with tqdm(total=total_combinations, desc="Soft K-Fold Tuning") as pbar:
            for n_tree in n_tree_list:
                for var_split in var_split_list:
                    for min_leaf_pop in min_leaf_pop_list:
                     print(f"Testing: Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                    fold_loss_list = []

                    for i in range(k):
                        try:
                            class_loss_list = []
                            
                            for class_id in classes:
                                # Use pre-sampled data instead of sampling each time
                                train_pixels = fold_class_samples[i][class_id]['train']
                                test_pixels = fold_class_samples[i][class_id]['test']
                                
                                #Random Forest Model, set to probability mode
                                clf = (ee.Classifier.smileRandomForest(
                                        numberOfTrees=n_tree,
                                        variablesPerSplit=var_split,
                                        minLeafPopulation=min_leaf_pop,
                                        seed=seed)
                                        .setOutputMode('PROBABILITY'))
                                model = clf.train(
                                    features=train_pixels,
                                    classProperty='binary',
                                    inputProperties=image.bandNames()
                                )
                                test_classified = test_pixels.classify(model)
                               # Extract true class labels and predicted probabilities
                                y_true = test_classified.aggregate_array('binary')
                                y_pred = test_classified.aggregate_array('classification')
                                paired = y_true.zip(y_pred).map(
                                        lambda xy: ee.Dictionary({
                                            'y_true': ee.List(xy).get(0),
                                            'y_pred': ee.List(xy).get(1)
                                        })
                                    )
                                # function to calculate log loss
                                def log_loss(pair_dict):
                                    pair_dict = ee.Dictionary(pair_dict)
                                    y = ee.Number(pair_dict.get('y_true'))
                                    p = ee.Number(pair_dict.get('y_pred'))
                                    #epsilon for numerical stability
                                    epsilon = 1e-15
                                    p_clip = p.max(epsilon).min(ee.Number(1).subtract(epsilon))
                                    # Log loss formula: -[y*log(p) + (1-y)*log(1-p)]
                                    loss = y.multiply(p_clip.log()).add(
                                        ee.Number(1).subtract(y).multiply(
                                            ee.Number(1).subtract(p_clip).log()
                                        )
                                    ).multiply(-1)
                                    return loss
                                loss_list = paired.map(log_loss)
                                avg_loss = ee.Number(loss_list.reduce(ee.Reducer.mean()))
                                class_loss = avg_loss.getInfo()
                                class_loss_list.append(class_loss)
                            
                            # Average loss across all classes for this fold
                            if class_loss_list:
                                fold_loss = sum(class_loss_list) / len(class_loss_list)
                                fold_loss_list.append(fold_loss)
                                print(f"Fold {i+1} Loss: {fold_loss:.6f}")

                        except Exception as e:
                            print(f"Failed for fold {i+1}")
                            print(f"Failed for Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                            print(f"Error: {e}")
                            continue
                    
                    # Calculate average loss across all folds for this parameter combination
                    if fold_loss_list:
                        avg_loss = sum(fold_loss_list) / len(fold_loss_list)
                        result.append({
                            'Number of Trees': n_tree,
                            'Variable Per Split': var_split,
                            'Minimum Leaf Population': min_leaf_pop,
                            'Average Model Cross Entropy Loss': avg_loss
                        })
                        print(f"Average Loss across folds: {avg_loss:.6f}")
                    else:
                        print(f"[WARNING] No valid loss scores for parameter combination Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                    
                    pbar.update(1)  # Update progress bar

        #convert the result into panda dataframe and view the best parameters
        if result:
            result_df = pd.DataFrame(result)
            result_df_sorted = result_df.sort_values(by='Average Model Cross Entropy Loss', ascending=True).reset_index(drop=True)
            
            print("\n" + "="*50)
            print("GRID SEARCH RESULTS")
            print("="*50)
            print("\nBest parameters (lowest loss):")
            print(result_df_sorted.iloc[0])
            print("\nTop 5 parameter combinations:")
            print(result_df_sorted.head())
            
            return result, result_df_sorted
        else:
            print("No successful parameter combinations found!")
            return [], pd.DataFrame()

# Final Classification
# Front-end: User can:
# * Accept optimal parameters from tuning
# * Or retain manual inputs Progress bars shown during training and map generation.
# Back-end: Trains the final Random Forest model using optimal parameters identified during tuning:
# * Bootstrap sampling: Creates diverse training subsets for each tree
# * Ensemble construction: Builds multiple decision trees with variation
# * Majority voting: Aggregates predictions across all trees
# Users can apply the recommended optimal parameters or use custom values.

class Generate_LULC:
    def __init__(self):
        """
        Perform classification to generate Land Cover Land Use Map. The parameters used in the classification should be the result of hyperparameter tuning
        """
        ee.Initialize()
        pass

    ############################# 1. Multiclass Classification ###########################
    # Multiclass Classification
    # Back-end: Performs direct multiclass classification where each pixel is assigned to a single LULC class.
    # Related Function(s): multiclass_classification() implements hard classification using the trained Random Forest model to generate the final LULC map.
    def multiclass_classification(self, training_data, class_property, image, ntrees = 100, 
                                  v_split = None, min_leaf = 1, seed=0):
        """
        Perform multiclass hard classification to generate land cover land use map
            Parameters:
            training data: ee.FeatureCollection, input sample data from feature extraction function (must contain pixel value)
            class_property (str): Column name contain land cover class id
            ntrees (int): Number of trees (user should input the best parammeter from parameter optimization)
            v_split (int): Variables per split (default = sqrt(#covariates)). (user should input the best parammeter from parameter optimization)
            min_leaf (int): Minimum leaf population. (user should input the best parammeter from parameter optimization)
            seed (int): Random seed.
        returns:
        ee.Image contain hard multiclass classification
        """
   # parameters and input valdiation
        if not isinstance(training_data, ee.FeatureCollection):
            raise ValueError("training_data must be an ee.FeatureCollection")
        if not isinstance(image, ee.Image):
            raise ValueError("image must be an ee.Image")
        #if for some reason var split is not specified, used 
        if v_split is None:
            v_split = ee.Number(image.bandNames().size()).sqrt().ceil()
        #Random Forest model
        clf = ee.Classifier.smileRandomForest(
                numberOfTrees=ntrees, 
                variablesPerSplit=v_split,
                minLeafPopulation=min_leaf,
                seed=seed)
        model = clf.train(
            features=training_data,
            classProperty=class_property,
            inputProperties=image.bandNames()
        )
        #Implement the trained model to classify the whole imagery
        multiclass = image.classify(model)
        return multiclass
     ############################# 1. One-vs-rest (OVR) binary Classification ###########################
    # OVR Classification
    # Back-end: Implements One-vs-Rest strategy, generating probability layers for each class and creating final classification via maximum probability.
    # Related Function(s): ovr_classification() creates probability surfaces for all LULC classes and produces final classification using argmax, providing confidence layers for each class.
    def ovr_classification(self, training_data, class_property, image, include_final_map=True,
                                ntrees = 100, v_split = None, min_leaf =1, seed=0, probability_scale = 100):
        """
        Implementation of one-vs-rest binary classification approach for multi-class land cover classification, similar to the work of
        Saah et al 2020. This function create probability layer stack for each land cover class. The final land cover map is created using
        maximum probability, via Argmax

        Parameters
            training_data (ee.FeatureCollection): The data which already have a pixel value from input covariates
            class_property (str): Column name contain land cover class id
            image (ee.Image): Image data
            covariates (list): covariates names
            ntrees (int): Number of trees (user should input the best parammeter from parameter optimization)
            v_split (int): Variables per split (default = sqrt(#covariates)). (user should input the best parammeter from parameter optimization)
            min_leaf (int): Minimum leaf population. (user should input the best parammeter from parameter optimization)
            seed (int): Random seed.
            probability scale = used to scaled up the probability layer

        Returns:
            ee.Image: Stacked probability bands + final classified map.
        """
        # parameters and input valdiation
        if not isinstance(training_data, ee.FeatureCollection):
            raise ValueError("training_data must be an ee.FeatureCollection")
        if not isinstance(image, ee.Image):
            raise ValueError("image must be an ee.Image")
        #if for some reason var split is not specified, used 
        if v_split is None:
            v_split = ee.Number(image.bandNames().size()).sqrt().ceil()
        
        # Get distinct classes ID from the training data. It should be noted that unique ID should be in integer, since 
        # float types tend to resulted in error during the band naming process 
        class_list = training_data.aggregate_array(class_property).distinct()
        
        #Define how to train one vs rest classification and map them all across the class
        def per_class(class_id):
            class_id = ee.Number(class_id)
            #Creating a binary features, 1 for a certain class and 0 for other (forest = 1, other = 0)
            binary_train = training_data.map(lambda ft: ft.set('binary', ee.Algorithms.If(
                            ee.Number(ft.get(class_property)).eq(class_id), 1, 0
                                )
                            ))
            #Build random forest classifiers, setting the outputmode to 'probability'. The probability mode will resulted in
            #one binary classification for each class. This give flexibility in modifying the final weight for the final land cover
            #multiprobability resulted in less flexibility in modifying the class weight
            #(the parameters required tuning)
            classifier = ee.Classifier.smileRandomForest(
                numberOfTrees=ntrees, 
                variablesPerSplit=v_split,
                minLeafPopulation=min_leaf,
                seed=seed
            ).setOutputMode("PROBABILITY")
            #Train the model
            trained = classifier.train(
                features=binary_train,
                classProperty="binary",
                inputProperties=image.bandNames()
            )
            # Apply to the image and get the probability layer
            # (probability 1 represent the confidence of a pixel belonging to target class)
            prob_img = image.classify(trained).select(['probability_1'])
            #scaled and convert to byte
            prob_img = prob_img.multiply(probability_scale).round().byte()
            #rename the bands
            #Ensure class_id is integer. 
            class_id_str = class_id.int().format()
            band_name = ee.String ('prob_').cat(class_id_str)

            return prob_img.rename(band_name)
        # Map over classes to get probability bands
        prob_imgs = class_list.map(per_class)
        prob_imgcol = ee.ImageCollection(prob_imgs)
        prob_stack = prob_imgcol.toBands()

        #if final map  is not needed, the functin will return prob bands only
        if not include_final_map:
            return prob_stack
        #final map creation using argmax
        print('Creating final classification using argmax')
        class_ids = ee.List(class_list)
        #find the mad prob in each band for each pixel
        #use an index image (0-based) indicating which class has highest probability
        max_prob_index = prob_stack.toArray().arrayArgmax().arrayGet(0)

        #map the index to actual ID
        final_lc = max_prob_index.remap(ee.List.sequence(0, class_ids.size().subtract(1)),
                                        class_ids).rename('classification')
        #calculate confidence layer
        max_confidence = prob_stack.toArray().arrayReduce(ee.Reducer.max(), [0]).arrayGet([0]).rename('confidence')
        #stack the final map and confidence
        stacked = prob_stack.addBands([final_lc, max_confidence])
        return stacked

# Review Classification Results
# Front-end:
# 1. User examines the generated LULC map and model performance
# 2. If satisfied, proceeds to Module 7 for thematic accuracy assessment
# 3. If unsatisfied, returns to parameter tuning or previous modules
# Back-end:
# 1. Displays optimal parameters and model performance metrics
# 2. Provides recommendations for improvement if results are unsatisfactory
# 3. Stores trained model and classification results for Module 7