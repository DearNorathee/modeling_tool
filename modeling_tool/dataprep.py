import inspect
from typing import Union, Dict, List, Any, Tuple, Literal
import pandas as pd

def upsampling(
        X_df: pd.DataFrame, 
        y_series: pd.Series, 
        strategy: Literal["equal"] | Dict[Union[str, int, float], float]  = 'equal', 
        random_state: int = 1, 
        verbose: int = 1,
        concat: bool = True
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:

    import pandas as pd
    import numpy as np
    from math import ceil

    # v03, Solo from o1 ChatGPT as of Nov, 6, 2024
    """
    Perform manual upsampling on a dataset to balance class distribution according to a specified strategy.

    Parameters:
    X_df (pd.DataFrame): DataFrame containing the feature set.
    y_series (pd.Series): Series containing the target variable with class labels.
    strategy (str or dict): If 'equal', all classes are upsampled to the same number as the majority class.
        If a dict, each class is upsampled to match a specified proportion.
    random_state (int): The seed used by the random number generator.
    verbose (int): 
        0 print nothing
        1 print before & after upsampling
    concat (bool): 
        If True, returns a single DataFrame combining X and y.
        If False, returns a tuple (X, y) as separate objects.

    Returns:
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        - If concat is True: Returns a DataFrame combining X and y.
        - If concat is False: Returns a tuple (X_train_oversampled, y_train_oversampled).
    """
    
    if not isinstance(y_series, pd.Series):
        raise Exception(f"Make sure that y_series is pd.Series type. Currently it's {type(y_series)}")

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
        # Ensure that the keys in strategy match the labels in y_series
        labels_set = set(labels)
        strategy_keys_set = set(strategy.keys())
        if labels_set != strategy_keys_set:
            missing_in_strategy = labels_set - strategy_keys_set
            extra_in_strategy = strategy_keys_set - labels_set
            error_message = "The keys in 'strategy' do not match the labels in 'y_series'."
            if missing_in_strategy:
                error_message += f" Missing labels in strategy: {missing_in_strategy}."
            if extra_in_strategy:
                error_message += f" Labels in strategy not present in y_series: {extra_in_strategy}."
            # Add an example of a valid strategy using the labels from y_series
            example_strategy = {label: round(1.0 / len(labels), 2) for label in labels}
            error_message += f" An example of a valid strategy is: {example_strategy}."
            raise ValueError(error_message)
        
        total_proportion = sum(strategy.values())
        normalized_strategy = {label: strategy[label] / total_proportion for label in labels}
        t_candidates = {}
        for label in labels:
            n_c = value_counts[label]
            p_c = normalized_strategy[label]
            if p_c > 0:
                t_candidate = n_c / p_c
            else:
                t_candidate = n_c  # If p_c is zero, t_candidate doesn't affect t
            t_candidates[label] = t_candidate
        t = max(t_candidates.values())
        target_counts = {}
        for label in labels:
            p_c = normalized_strategy[label]
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
    
    if concat:
        # Combine X and y into a single DataFrame
        X_y_combined = X_train_oversampled.copy()
        y_name = y_series.name if y_series.name is not None else 'target'
        X_y_combined[y_name] = y_train_oversampled
        return X_y_combined
    else:
        return (X_train_oversampled, y_train_oversampled)

def smote_nc_upsampling(
        X_df: pd.DataFrame, 
        y_series: pd.Series, 
        strategy: Literal["equal"] | Dict[Union[str, int, float], float] = 'equal', 
        random_state: int = 1, 
        verbose: int = 1,
        concat: bool = True
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    
    import pandas as pd
    import numpy as np
    from imblearn.over_sampling import SMOTE, SMOTENC

    # still has bbug with strategy as dict
    """
    Perform SMOTE upsampling on a dataset to balance class distribution according to a specified strategy.

    Parameters:
    X_df (pd.DataFrame): DataFrame containing the feature set.
    y_series (pd.Series): Series containing the target variable with class labels.
    strategy (str or dict): If 'equal', all classes are upsampled to the same number as the majority class.
        If a dict, each class is upsampled to match a specified proportion.
    random_state (int): The seed used by the random number generator.
    verbose (int): 
        0 print nothing
        1 print before & after upsampling
    concat (bool): 
        If True, returns a single DataFrame combining X and y.
        If False, returns a tuple (X_train_oversampled, y_train_oversampled).

    Returns:
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        - If concat is True: Returns a DataFrame combining X and y.
        - If concat is False: Returns a tuple (X_train_oversampled, y_train_oversampled).
    """
    
    if not isinstance(y_series, pd.Series):
        raise Exception(f"Make sure that y_series is pd.Series type. Currently it's {type(y_series)}")
    
    if verbose == 1:
        print("Before upsampling: ")
        print(y_series.value_counts(), "\n")
    
    # Determine the sampling strategy for SMOTE
    if strategy == 'equal':
        sampling_strategy = 'not majority'
    elif isinstance(strategy, dict):
        labels_set = set(y_series.unique())
        strategy_keys_set = set(strategy.keys())
        if labels_set != strategy_keys_set:
            missing_in_strategy = labels_set - strategy_keys_set
            extra_in_strategy = strategy_keys_set - labels_set
            error_message = "The keys in 'strategy' do not match the labels in 'y_series'."
            if missing_in_strategy:
                error_message += f" Missing labels in strategy: {missing_in_strategy}."
            if extra_in_strategy:
                error_message += f" Labels in strategy not present in y_series: {extra_in_strategy}."
            # Add an example of a valid strategy using the labels from y_series
            example_strategy = {label: round(1.0 / len(labels_set), 2) for label in labels_set}
            error_message += f" An example of a valid strategy is: {example_strategy}."
            raise ValueError(error_message)
        
        total_proportion = sum(strategy.values())
        normalized_strategy = {label: strategy[label] / total_proportion for label in labels_set}
        sampling_strategy = {label: normalized_strategy[label] for label in labels_set if normalized_strategy[label] > 0}
    else:
        raise ValueError("Strategy must be 'equal' or a dict of class proportions")
    
    categorical_columns = X_df.select_dtypes(include='object').columns
    # Get the indices of categorical columns
    categorical_indices = [X_df.columns.get_loc(col) for col in categorical_columns]

    # Instantiate SMOTE with the specified random state
    smote = SMOTENC(categorical_features = categorical_indices,sampling_strategy=sampling_strategy, random_state=random_state)
    
    # Fit and resample the data
    X_resampled, y_resampled = smote.fit_resample(X_df, y_series)
    
    if verbose == 1:
        print("After upsampling: ")
        print(pd.Series(y_resampled).value_counts(), "\n")
    
    if concat:
        # Combine X and y into a single DataFrame
        X_y_combined = pd.DataFrame(X_resampled, columns=X_df.columns)
        y_name = y_series.name if y_series.name is not None else 'target'
        X_y_combined[y_name] = y_resampled
        return X_y_combined
    else:
        return (pd.DataFrame(X_resampled, columns=X_df.columns), pd.Series(y_resampled))


# prevent showing many objects from import when importing this module
# from typing import *
del Union
del Dict
del List
