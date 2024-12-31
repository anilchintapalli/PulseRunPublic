import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality



def filter_incomplete_runs(df):
    '''
    This function preprocesses the running data by filtering out the unnecessary columns.
    It then converts units as necessary to appeal to an American audience

    Args:
        df: pandas dataframe that contains the user's running data

    Returns:
        pandas dataframe: the filtered and transformed dataframe 
    
    '''
    
    required_columns = ['distance','averageSpeed','elevationGain','elevationLoss','calories',
                        'averageHR','maxHR','activityTrainingLoad','avgPower','avgVerticalOscillation',
                        'avgGroundContactTime','avgStrideLength','averageRunningCadenceInStepsPerMinute']
    
    
    filtered_df = df.dropna(subset=required_columns)[required_columns]
    filtered_df['distance'] = filtered_df['distance'] / 1609.344
    filtered_df['averagePace'] = ((1 / (filtered_df['averageSpeed'] / 1609.344))/60)
    filtered_df = filtered_df.drop(columns=['averageSpeed'])
    filtered_df['elevationGain'] = filtered_df['elevationGain'] * 3.28084
    filtered_df['elevationLoss'] = filtered_df['elevationLoss'] * 3.28084 
    
    return filtered_df

def remove_outliers(df, threshold=1.5):
    '''
    uses the interquartile range to remove statistical outliers from the data before 
    generating synthetic data: 
    https://medium.com/@pp1222001/outlier-detection-and-removal-using-the-iqr-method-6fab2954315d

    Args:
        df: pandas dataframe that contains the running metric (after it has been cleaned)
        threfold: mutliplied by IQR to find outlier bounds (float; 1.5 default)
    
        Returns:
            pandas dataframe with the outliers removed
    
    '''
    
    outlier_df = df.copy()
    outlier_stats = {}
    
    # remove the statistical outliers from each indicated column
    for column in outlier_df.columns:
        
        Q1 = outlier_df[column].quantile(0.25)
        Q3 = outlier_df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        
        lower_bound = Q1 - (threshold * IQR)
        upper_bound = Q3 + (threshold * IQR)
        
        
        outliers_before = len(outlier_df)
        
        
        outlier_df = outlier_df[
            (outlier_df[column] >= lower_bound) & 
            (outlier_df[column] <= upper_bound)
        ]
        
        
        outliers_removed = outliers_before - len(outlier_df)
        outlier_stats[column] = {
            'total_outliers': outliers_removed,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'percent_removed': (outliers_removed / outliers_before) * 100
        }
    
    
    print("Outlier Removal Statistics:")
    for column, stats in outlier_stats.items():
        print(f"{column}:")
        print(f"  Outliers removed: {stats['total_outliers']}")
        print(f"  Lower bound: {stats['lower_bound']:.2f}")
        print(f"  Upper bound: {stats['upper_bound']:.2f}")
        print(f"  Percent removed: {stats['percent_removed']:.2f}%")
    
    return outlier_df




def generate_synthetic_data(df):
    '''
    generates synthetic running data using SDV (1000 rows)

    Args:
        df: pandas dataframe with the filter running metrics (& with outliers removed)

    Returns:
        New dataframe with 1000 rows of synthetical generated data

    
    '''

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)  

    sdv_model = GaussianCopulaSynthesizer(metadata)


    sdv_model.fit(df)


    synthetic_data = sdv_model.sample(num_rows=1000)

    quality_report = evaluate_quality(
        real_data=df,
        synthetic_data=synthetic_data,
        metadata=metadata)
    print(quality_report)
    return synthetic_data
