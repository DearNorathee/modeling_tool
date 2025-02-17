import pandas as pd
import numpy as np
from typing import Union

def plot_cat_metric_rank(y_prob, y_label, positive_label=1):
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm
    """
    Plot metrics (Precision, Recall, F1 Score, AUC, Prob Threshold) against the number of predictions.

    Parameters:
    y_prob (array-like): Predicted probabilities for the positive class.
    y_label (array-like): True labels of the test set.
    positive_label (int, optional): The label representing the positive class. Default is 1.
    """
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import numpy as np

    # Convert y_label to binary (0 and 1)
    y_label_num = y_label.copy()
    if isinstance(y_label, (pd.Series, pd.DataFrame)):
        y_label_num = y_label_num.apply(lambda x: 1 if x == positive_label else 0)
    elif isinstance(y_label, np.ndarray):
        y_label_num = np.where(y_label == positive_label, 1, 0)

    # Create a DataFrame with y_prob and y_label_num
    df = pd.DataFrame({'y_prob': y_prob, 'y_label': y_label_num})

    # Sort by predicted probabilities in descending order
    df_sorted = df.sort_values(by='y_prob', ascending=False).reset_index(drop=True)

    # Initialize lists to store metrics
    precisions, recalls, f1_scores, aucs, thresholds = [], [], [], [], []

    for i in tqdm(range(1, len(df_sorted) + 1), desc="Creating plots..."):
        # Get the top i predictions
        df_top = df_sorted.head(i)

        # Calculate metrics
        precision = precision_score(df_top['y_label'], [1] * i, pos_label=1, zero_division=0)
        recall = recall_score(df_sorted['y_label'], [1 if idx < i else 0 for idx in range(len(df_sorted))], pos_label=1)
        f1 = f1_score(df_sorted['y_label'], [1 if idx < i else 0 for idx in range(len(df_sorted))], pos_label=1)
        auc = roc_auc_score(df_sorted['y_label'], df_sorted['y_prob'])
        threshold = df_top['y_prob'].iloc[-1]

        # Append metrics to lists
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        aucs.append(auc)
        thresholds.append(threshold)

    # Plot metrics
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(df_sorted) + 1), precisions, label='Precision', color='b')
    plt.plot(range(1, len(df_sorted) + 1), recalls, label='Recall', color='g')
    plt.plot(range(1, len(df_sorted) + 1), f1_scores, label='F1 Score', color='r')
    plt.plot(range(1, len(df_sorted) + 1), aucs, label='AUC', color='c')
    plt.plot(range(1, len(df_sorted) + 1), thresholds, label='Prob Threshold', color='m')

    # Add labels and legend
    plt.xlabel('Number of Predictions')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs Number of Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def cal_rank_cat_metric(
        y_label
        ,y_prob
        ,positive_label=1
        ,dev_mode:int = False
        ) -> pd.DataFrame:
    from tqdm.auto import tqdm
    """
    Calculate classification metrics for each rank based on sorted probabilities.

    Parameters:
    y_label (array-like): True labels of the test set.
    y_prob (array-like): Predicted probabilities for the positive class.
    positive_label (int): The label representing the positive class.

    Returns:
    pd.DataFrame: DataFrame containing N_predictions, prob, label, TP, TN, FP, FN, precision, recall, f1, auc, accuracy.
    """
    # Convert y_label to binary (0 and 1)
    y_label_num = np.array([1 if label == positive_label else 0 for label in y_label])
    y_prob = np.array(y_prob)

    # Create a DataFrame with y_prob and y_label_num and sort by probabilities
    df = pd.DataFrame({'prob': y_prob, 'label': y_label_num})
    df_sorted = df.sort_values(by='prob', ascending=False).reset_index(drop=True)

    # Initialize lists to store metrics
    metrics = []

    # Initialize counts for TP, TN, FP, FN
    TP = FP = TN = FN = 0

    for i in (tqdm( range(len(df_sorted)), desc = "Calculating metrics...", leave=True)):
        label = df_sorted.loc[i, 'label']
        if label == 1:
            TP += 1
        else:
            FP += 1

        # Update TN and FN based on remaining labels
        FN = df_sorted['label'].sum() - TP
        TN = len(df_sorted) - (TP + FP + FN)

        # Calculate metrics manually
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        # auc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

        metrics.append({
            'N_predictions': i + 1,
            'prob': df_sorted.loc[i, 'prob'],
            'label': label,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            # 'auc': auc,
            'accuracy': accuracy
        })

    # Convert the metrics list to a DataFrame for display
    metrics_df = pd.DataFrame(metrics)
    if not dev_mode:
        metrics_df = metrics_df.drop(columns = ["TP","TN","FP","FN"])
        
    return metrics_df


def cat_threshold_plot(metric_df: pd.DataFrame, bar_alpha = 0.3):
    import matplotlib.pyplot as plt
    # v03
    """
    Plot metrics (Precision, Recall, F1 Score, ROC AUC) against different thresholds.

    Parameters:
    metric_df (pd.DataFrame): DataFrame containing the metrics for each threshold.
    """
    # Plotting the metrics
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot each metric as a line graph
    ax1.plot(metric_df['Threshold'], metric_df['Precision'], label='Precision', marker='o')
    ax1.plot(metric_df['Threshold'], metric_df['Recall'], label='Recall', marker='o')
    ax1.plot(metric_df['Threshold'], metric_df['F1 Score'], label='F1 Score', marker='o')
    ax1.plot(metric_df['Threshold'], metric_df['AUC'], label='ROC AUC', marker='o')

    # Set labels for the line plot
    ax1.set_xlabel('Probability Threshold')
    ax1.set_xticks(metric_df['Threshold'])
    ax1.set_ylabel('Metric Value')
    ax1.set_ylim(0, 1.1)
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_title('Probability Threshold Plot')
    ax1.legend(loc='upper right')  # Move legend to the upper right
    ax1.grid(True)

    # Create a second y-axis to plot the number of positive predictions as a bar plot
    ax2 = ax1.twinx()
    auto_bar_width = metric_df['Threshold'].diff().mean() * 0.8
    

    ax2.bar(metric_df['Threshold'], metric_df['n_predictions'], alpha=bar_alpha, width=auto_bar_width, label='n_predictions')
    ax2.set_ylabel('Number of Positive Predictions (n_predictions)')

    # Add labels on top of the bars
    for idx, val in enumerate(metric_df['n_predictions']):
        ax2.text(metric_df['Threshold'][idx], val + 1, str(val), ha='center', va='bottom', fontsize=13,color='blue')

    # Show the plot
    fig.tight_layout()
    plt.show()


def cal_cat_metrics(
        pred_actual: Union[list, pd.Series]
        ,pred_prob: Union[list, pd.Series]
        ,thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ,positive_class: Union[list,str] = 1) -> pd.DataFrame:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    """
    Calculate classification metrics for different thresholds.

    Parameters:
    pred_label (array-like): True labels of the test set.
    pred_prob (array-like): Predicted probabilities for the positive class.
    thresholds (array-like): List or range of thresholds to evaluate.
    positive_class (int): The label representing the positive class.

    Returns:
    pd.DataFrame: DataFrame containing metrics for each threshold.
    """
    # Store the metrics in a list for easy conversion to DataFrame
    metrics = []
    pred_actual_num = pred_actual.copy()
    
    if isinstance(pred_actual, (pd.Series,pd.DataFrame)):
        pred_actual_num = pred_actual_num.apply(lambda x: 1 if x == positive_class else 0)
    elif isinstance(pred_actual, np.ndarray):
        pred_actual_num = np.where(pred_actual == positive_class, 1, 0)

    for threshold in thresholds:
        # Apply the threshold to get binary predictions
        y_pred = (pred_prob > threshold).astype(int)
        n_predictions = y_pred.sum()
        
        # Calculate metrics
        accuracy = accuracy_score(pred_actual_num, y_pred)
        precision = precision_score(pred_actual_num, y_pred, zero_division=0)
        recall = recall_score(pred_actual_num, y_pred)
        f1 = f1_score(pred_actual_num, y_pred)
        auc = roc_auc_score(pred_actual_num, pred_prob)  # AUC is independent of threshold

        metrics.append({
            'Threshold': threshold,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC': auc,
            'n_predictions':n_predictions
        })

    # Convert the metrics list to a DataFrame for display
    metrics_df = pd.DataFrame(metrics)
    return metrics_df