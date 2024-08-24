import inspect
from typing import Union, Dict, List
import pandas as pd

def upsampling(
        X_df: pd.DataFrame, 
        y_series: pd.Series, 
        strategy: Union[str, Dict[str, float]] = 'equal', 
        random_state: int = 1, 
        verbose: int = 1) -> List:
    # medium tested seem to work now
    # solo from ChatGPT
    import pandas as pd
    import numpy as np
    """
    Perform manual upsampling on a dataset to balance class distribution according to a specified strategy.

    Parameters:
    X_df (pd.DataFrame): DataFrame containing the feature set.
    y_df (pd.Series): Series containing the target variable with class labels.
    strategy (str or dict): If 'equal', all classes are upsampled to the same number as the majority class. If a dict, each class is upsampled to match a specified proportion.
    random_state (int): The seed used by the random number generator.
    verbose (int): 
        0 print nothing
        1 print before & after upsampling

    Returns:
    list: Contains two elements:
        - pd.DataFrame: The upsampled feature DataFrame.
        - pd.Series: The upsampled target Series.
    """
    
    if not isinstance(y_series,pd.Series):
        raise Exception(f"Make sure that y_series is pd.Series type. currently it's {type(y_series)}")

    np.random.seed(random_state)

    if verbose == 1:
        print("Before upsampling: ")
        print(y_series.value_counts(), "\n")
    
    # Determine the majority class and its count
    if strategy == 'equal':
        majority_count = y_series.value_counts().max()
        sample_proportions = {label: majority_count for label in y_series.unique()}
    elif isinstance(strategy, dict):
        total_proportion = sum(strategy.values())
        sample_proportions = {label: int((strategy.get(label, 0) / total_proportion) * len(y_series)) for label in y_series.unique()}
    
    # Initialize the upsampled DataFrames
    X_train_oversampled = pd.DataFrame()
    y_train_oversampled = pd.Series()

    # Perform manual oversampling for each class
    for label, target_count in sample_proportions.items():
        indices = y_series[y_series == label].index
        if len(indices) == 0:
            continue
        if target_count <= len(indices):
            sampled_indices = np.random.choice(indices, target_count, replace=False)
        else:
            sampled_indices = np.random.choice(indices, target_count, replace=True)
        X_train_oversampled = pd.concat([X_train_oversampled, X_df.loc[sampled_indices]], axis=0)
        y_train_oversampled = pd.concat([y_train_oversampled, y_series.loc[sampled_indices]])

    # Reset index to avoid duplicate indices
    X_train_oversampled.reset_index(drop=True, inplace=True)
    y_train_oversampled.reset_index(drop=True, inplace=True)

    if verbose == 1:
        print("After upsampling: ")
        print(y_train_oversampled.value_counts(), "\n")
    
    return [X_train_oversampled, y_train_oversampled]


# prevent showing many objects from import when importing this module
# from typing import *
del Union
del Dict
del List
