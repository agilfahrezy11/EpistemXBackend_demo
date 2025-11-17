import ee
import pandas as pd
ee.Initialize()

class Evaluate_LULC:
    def __init__(self):
        pass

    def thematic_assessment(self, lcmap, validation_data, class_property,
                            region=None, scale=10, return_per_class = True):
        """
        Evaluate the thematic accuracy of land cover land use map wih several accuracy metric:
        overall accuracy
        F1-score
        Geometric mean
        Per-class accuracy
        balanced accuracy score
        
        """
        if region is None:
            validation_sample = lcmap.select('classification').sampleRegions(
                collection = validation_data,
                properties = [class_property],
                scale = scale,
                geometries = False,
                tileScale = 4
            )
        else:
            validation_sample = lcmap.select('classification').sampleRegions(
                collection = validation_data.filterBounds(region),
                properties = [class_property],
                scale = scale,
                tileScale = 4
            )
        #create a confuction matrix
        confusion_matrix = validation_sample.errorMatrix(class_property, 'classification')
        #basic metric calculation:
        oa = confusion_matrix.accuracy()
        kappa = confusion_matrix.kappa()
        #calculate per class accuracy
        class_order = confusion_matrix.order()
        matrix_array = confusion_matrix.array()
        def per_class_acc():
            """
            calculate precision, recal, and f1 per class
            """
            n_class = matrix_array.length().get(0)
            def per_class_calc(i):
                i = ee.Number(i)
                #TP = True Positive
                tp = matrix_array.get([i, i])
                #FP = False positive
                col_sum = matrix_array.slice(0,0,-1).slice(1, i, i.add(1)).reduce(ee.Reuducer.sum(), [0])
                fp = col_sum.get([0,0]).subtract(tp)
                #fn = false negatives
                row_sum = matrix_array.slice(0, i, i.add(1)).slice(1,0,-1).reduce(ee.Reducer.sum(), [1])
                fn = row_sum.get([0,0]).subtract(tp)
                #calculate precision (user accuracy), recall (producer accuracy), and f1-score
                precision = ee.Number(tp).divide(ee.Number(tp).add(ee.Number(fp)))
                recall = ee.Number(tp).divide(ee.Number(tp).add(ee.Number(fn)))
                #f1 score with zero division handling
                f1 = ee.Algorithms.If(precision.add(recall).eq(0), 0,
                                    precision.multiply(recall).multiply(2).divide(precision.add(recall))
                                    )
                return ee.Dictionary({
                    'class_ID': class_order.get(i),
                    'Preicison/User Accuracy': precision, 
                    'Recall/Producer Accuracy': recall,
                    'F1 Score': f1,
                    'True Positive': tp,
                    'False Positive': fp,
                    'False Negative': fn
                })
            #map all classes
            class_index = ee.List.sequence(0, n_class.subtract(1))
            per_class_result = class_index.map(per_class_calc)
            return per_class_result
        
        #now calculate the metric:
        per_class_metrics = per_class_acc()
            # Calculate macro-averaged metrics
        def calculate_macro_metrics():
            """Calculate macro-averaged F1, Precision, Recall"""
            # Extract individual metric lists
            precision_values = per_class_metrics.map(lambda x: ee.Dictionary(x).get('precision'))
            recall_values = per_class_metrics.map(lambda x: ee.Dictionary(x).get('recall'))
            f1_values = per_class_metrics.map(lambda x: ee.Dictionary(x).get('f1_score'))
            # Calculate means
            macro_precision = ee.Array(precision_values).reduce(ee.Reducer.mean(), [0]).get([0])
            macro_recall = ee.Array(recall_values).reduce(ee.Reducer.mean(), [0]).get([0])
            macro_f1 = ee.Array(f1_values).reduce(ee.Reducer.mean(), [0]).get([0])
            return {
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1
            }
        # Calculate micro-averaged metrics
        def calculate_micro_metrics():
            """Calculate micro-averaged F1, Precision, Recall"""
            # Sum all TP, FP, FN across classes
            total_tp = ee.Array(per_class_metrics.map(lambda x: ee.Dictionary(x).get('tp'))).reduce(ee.Reducer.sum(), [0]).get([0])
            total_fp = ee.Array(per_class_metrics.map(lambda x: ee.Dictionary(x).get('fp'))).reduce(ee.Reducer.sum(), [0]).get([0])
            total_fn = ee.Array(per_class_metrics.map(lambda x: ee.Dictionary(x).get('fn'))).reduce(ee.Reducer.sum(), [0]).get([0])
            # Calculate micro metrics
            micro_precision = ee.Number(total_tp).divide(ee.Number(total_tp).add(ee.Number(total_fp)))
            micro_recall = ee.Number(total_tp).divide(ee.Number(total_tp).add(ee.Number(total_fn)))
            micro_f1 = micro_precision.multiply(micro_recall).multiply(2).divide(micro_precision.add(micro_recall))
            return {
                'micro_precision': micro_precision,
                'micro_recall': micro_recall,
                'micro_f1': micro_f1
            }
        # Calculate Geometric Mean
        def calculate_geometric_mean():
            """Calculate Geometric Mean of per-class recalls (sensitivities)"""
            
            recall_values = per_class_metrics.map(lambda x: ee.Dictionary(x).get('recall'))
            # Calculate geometric mean: (r1 * r2 * ... * rn)^(1/n)
            # Using log transform: exp(mean(log(recalls)))
            log_recalls = ee.Array(recall_values).log()
            mean_log_recall = log_recalls.reduce(ee.Reducer.mean(), [0]).get([0])
            geometric_mean = ee.Number(mean_log_recall).exp()
            return geometric_mean
        
        # Calculate Balanced Accuracy
        # Should be reconsidered since it could be redundant with OA
        def calculate_balanced_accuracy():
            """Calculate Balanced Accuracy (macro-averaged recall)"""
            recall_values = per_class_metrics.map(lambda x: ee.Dictionary(x).get('recall'))
            balanced_accuracy = ee.Array(recall_values).reduce(ee.Reducer.mean(), [0]).get([0])
            return balanced_accuracy
        
        # Get all calculated metrics
        macro_metrics = calculate_macro_metrics()
        micro_metrics = calculate_micro_metrics()
        geometric_mean = calculate_geometric_mean()
        balanced_accuracy = calculate_balanced_accuracy()
        # append the result into a dictionary
        results = {
            'confusion_matrix': confusion_matrix,
            'class_order': class_order,
            'overall accuracy': oa,
            'balanced accuracy': balanced_accuracy,
            'kappa': kappa,
            'macro_f1': macro_metrics['macro_f1'],
            'macro_precision': macro_metrics['macro_precision'], 
            'macro_recall': macro_metrics['macro_recall'],
            'micro_f1': micro_metrics['micro_f1'],
            'micro_precision': micro_metrics['micro_precision'],
            'micro_recall': micro_metrics['micro_recall'],
            'geometric_mean': geometric_mean,
        }
        if return_per_class: 
            results['per_class_metric'] = per_class_metrics
        return results
    ############################# 8. Printing the accuracy metrics ###########################
    # Display and Review Accuracy Results
    # Front-end: 1. User reviews the test results, including the Confusion Matrix and all accuracy metrics displayed by print_metrics(). 
    # 2. If satisfied, the process completes. 
    # 3. If unsatisfied, the user is prompted to: * Go back to @sec-module-3 to check and clarify the Reference Data. * Go back to @sec-module-6 to refine the hyperparameters or try a different classification approach.
    # Back-end:
    # * Formats and displays comprehensive accuracy report
    # * Identifies potential issues based on error patterns
    # * Provides specific recommendations for improvement
    # Related Functions:
    # * print_metrics() - displays formatted accuracy results with interpretations
    def print_metrics (self, evaluation, class_names = None):
        
        print("CLASSIFICATION THEMATIC EVALUATION RESULT\n")
        #overall metrics
        print('Overall Metrics:')
        print(f'Overall Accuracy:{evaluation["overall_accuracy"].getInfo():.4f}')
        print(f'Balanced Accuracy: {evaluation["balanced_accuracy"].getInfo():.4f}')
        print(f'Kappa Coefficient:{evaluation["kappa"].getInfo():.4f}')
        print(f'Geometric Mean: {evaluation["geometric_mean"].getInfo():.4f}')

        #Aggregate Metric
        print('Aggregate Metric:')
        print(f"Macro F-1 Score: {evaluation['macro_f1'].getInfo():.4f}")
        print(f"Micro F1-Score: {evaluation['micro_f1'].getInfo():.4f}")
        print(f"Macro Precision: {evaluation['macro_precision'].getInfo():.4f}")
        print(f"Macro Recall: {evaluation['macro_recall'].getInfo():.4f}")

        #per class if requested
        if 'per_class_metrics' in evaluation:
            print('PER-CLASS METRICS:')
            per_class = evaluation['per_class_metrics'].getInfo()
            print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
            print('-' * 45)
            for metrics in per_class:
                class_id = metrics['class_id']
                class_names = class_names.get(class_id, f"class_{class_id}") if class_names else f"class_{class_id}"
                print(f"{class_names:<10} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")
        
        print("\n=== METRIC INTERPRETATIONS ===")
        print("• Overall Accuracy: Can be misleading with imbalanced data")
        print("• Balanced Accuracy: Average of per-class recalls (better for imbalanced data)")
        print("• Macro F1: Unweighted average F1 across classes (treats all classes equally)")
        print("• Micro F1: Weighted by class frequency (dominated by frequent classes)")
        print("• Geometric Mean: Sensitive to poor performance on any class")
        print("• Kappa: Agreement beyond chance, accounts for class imbalance")    