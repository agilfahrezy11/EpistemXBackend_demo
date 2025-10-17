from src.module_helpers import init_gee
init_gee()
from scipy import stats
import numpy as np


class Accuracy_Assessment:
    def __init__(self):
        """
        Perform classification to generate Land Cover Land Use Map. The parameters used in the classification should be the result of hyperparameter tuning
        """
        pass
    #Function to calculate overall accuracy confidence interval
    def _calculate_accuracy_confidence_interval(self, n_correct, n_total, confidence=0.95):
        """
        Calculate confidence interval for overall accuracy using normal approximation.
        """
        if n_total == 0:
            return (0, 0)
        p = n_correct / n_total
        se = np.sqrt((p * (1 - p)) / n_total)
        z = stats.norm.ppf((1 + confidence) / 2)
        margin = z * se
        lower = max(0, p - margin)
        upper = min(1, p + margin)
        return lower, upper
    
    #Main Function to calculate the thematic accuracy 
    def thematic_assessment(self, lcmap, validation_data, class_property,
                            scale=30, confidence=0.95):
        """
        Perform thematic accuracy assessment on the resulting land cover data (categorical raster) using independent ground reference data.
        The approach use here similar to model evaluation procedure in module 6, with the main difference lies upon the data being tested.

        Parameters
            lcmap (ee.image): Categorical raster data from earth engine classifier. Must contain the band name: classification
            validation_data (ee.featureCollection): Ground refrence data containing class id and names, containing class id and names
            class_property (str):  Class property (column) contain unique class ID
            scale (str): spatial resolution of the land cover dat
            confidence (numeric): Confidence interval for overall accuracy

        Returns:
            accuracy metric (dict)
        
        """
        # quick check to make sure that the band names is correct
        if 'classification' not in lcmap.bandNames().getInfo():
            raise ValueError("Input land cover map must contain a band named 'classification'")
        #Sample the classified map to get the predicted lc data
        validation_sample = lcmap.select('classification').sampleRegions(
                collection=validation_data,
                properties=[class_property],
                scale=scale,
                geometries=False,
                tileScale=4
            )
        
        #Create confusion matrix
        confusion_matrix = validation_sample.errorMatrix(class_property, 'classification')
        
        #Extract the confusion matrix related information
        overall_accuracy = confusion_matrix.accuracy().getInfo()
        kappa = confusion_matrix.kappa().getInfo()
        #here still used earth engine terminology
        producers_accuracy_ls = confusion_matrix.producersAccuracy().getInfo()
        #here still used earth engine terminology
        consumers_accuracy_ls = confusion_matrix.consumersAccuracy().getInfo()
        #Extract the array of confusion matrix
        cm_array = confusion_matrix.getInfo()['array']
        #Flatten using numpy
        producers_accuracy = np.array(producers_accuracy_ls).flatten().tolist()
        consumers_accuracy = np.array(consumers_accuracy_ls).flatten().tolist()

        #Confidence interval calculation
        #Compute n_correct and n_total for CI
        n_correct = np.trace(np.array(cm_array))
        n_total = np.sum(np.array(cm_array))
        # Calculate 95% CI for overall accuracy
        ci_lower, ci_upper = self._calculate_accuracy_confidence_interval(n_correct, n_total, confidence)

        # Calculate F1 scores
        #create and empty dict for storing the result
        #use remote sensing terminology
        f1_scores = []
        for i in range(len(producers_accuracy)):
            producer_acc = producers_accuracy[i] #recall (machine learning terms)
            user_acc = consumers_accuracy[i] #precision (machine learning terms)
            #calculate each class f1 score first
            if producer_acc + user_acc > 0:
                f1 = 2 * (producer_acc * user_acc) / (producer_acc + user_acc)
            
            else:
                f1 = 0
            f1_scores.append(f1)        
        #Compile the accuracy metrics results
        accuracy_metrics = {
            'overall_accuracy': overall_accuracy,
            'kappa': kappa,
            'user_accuracy': consumers_accuracy,
            'producer_accuracy': producers_accuracy,
            'confusion_matrix': cm_array,
            'f1_scores': f1_scores,
            'overall_accuracy_ci': (ci_lower, ci_upper),
            'n_total': int(n_total)
        }
        
        return accuracy_metrics