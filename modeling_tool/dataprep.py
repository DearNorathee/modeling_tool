import inspect
from typing import Union, Dict, List, Any, Tuple
import pandas as pd

def upsampling(
        X_df: pd.DataFrame, 
        y_series: pd.Series, 
        strategy: Union[str, Dict[str|int|float, float]] = 'equal', 
        random_state: int = 1, 
        verbose: int = 1) -> tuple[pd.DataFrame,pd.DataFrame]:
    
    # v02: Solo by o1, as of Nov 6, 2024
    # medium tested
    import pandas as pd
    import numpy as np
    from math import ceil
    """
    Perform manual upsampling on a dataset to balance class distribution according to a specified strategy.

    Parameters:
    X_df (pd.DataFrame): DataFrame containing the feature set.
    y_series (pd.Series): Series containing the target variable with class labels.
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
    
    value_counts = y_series.value_counts()
    labels = y_series.unique()
    
    # Determine the target counts for each class
    if strategy == 'equal':
        majority_count = value_counts.max()
        target_counts = {label: majority_count for label in labels}
    elif isinstance(strategy, dict):
        total_proportion = sum(strategy.values())
        normalized_strategy = {label: strategy.get(label, 0) / total_proportion for label in labels}
        t_candidates = {}
        for label in labels:
            n_c = value_counts[label]
            p_c = normalized_strategy.get(label, 0)
            if p_c > 0:
                t_candidate = n_c / p_c
            else:
                t_candidate = n_c  # If p_c is zero, t_candidate doesn't affect t
            t_candidates[label] = t_candidate
        t = max(t_candidates.values())
        target_counts = {}
        for label in labels:
            p_c = normalized_strategy.get(label, 0)
            n_c = value_counts[label]
            if p_c > 0:
                target_count = max(n_c, ceil(t * p_c))
            else:
                target_count = n_c  # If p_c is zero, keep the original count
            target_counts[label] = target_count
    else:
        raise ValueError("Strategy must be 'equal' or a dict of class proportions")
    
    # Initialize the upsampled DataFrames
    X_train_oversampled = pd.DataFrame()
    y_train_oversampled = pd.Series(dtype=y_series.dtype)

    # Perform manual oversampling for each class
    for label, target_count in target_counts.items():
        indices = y_series[y_series == label].index
        if len(indices) == 0:
            continue
        replace = target_count > len(indices)
        sampled_indices = np.random.choice(indices, target_count, replace=replace)
        X_train_oversampled = pd.concat([X_train_oversampled, X_df.loc[sampled_indices]], axis=0)
        y_train_oversampled = pd.concat([y_train_oversampled, y_series.loc[sampled_indices]])

    # Reset index to avoid duplicate indices
    X_train_oversampled.reset_index(drop=True, inplace=True)
    y_train_oversampled.reset_index(drop=True, inplace=True)

    if verbose == 1:
        print("After upsampling: ")
        print(y_train_oversampled.value_counts(), "\n")
    
    return (X_train_oversampled, y_train_oversampled)



# prevent showing many objects from import when importing this module
# from typing import *
del Union
del Dict
del List
