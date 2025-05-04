import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from lifelines.utils import concordance_index

def normalize_zscore(df, feature_cols):
    df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / (df[feature_cols].std() + 1e-8)
    return df

def compute_chagas_weights(max_weight=np.inf):
    chagas_data = pd.read_excel('chagas_idades.xlsx')

    classe_1 = chagas_data[(chagas_data['Obito_MS'] == 1) & (chagas_data['Time'] < 5)]
    classe_0 = chagas_data[(chagas_data['Obito_MS'] == 0) | (chagas_data['Time'] == 5)]

    filtered_data = pd.concat([classe_1, classe_0])
    labels = np.where(filtered_data['Obito_MS'] == 1, 1, 0)

    class_counts = np.bincount(labels)
    weights = 1 / class_counts[labels]
    normalized_weights = weights / np.sum(weights) * len(labels)

    if max_weight < np.inf:
        normalized_weights = np.minimum(normalized_weights, max_weight)
        normalized_weights = normalized_weights / np.sum(normalized_weights) * len(labels)

    weights_dict = dict(zip(filtered_data['ID'].astype(str), normalized_weights))

    with open('chagas_weights.json', 'w') as json_file:
        json.dump(weights_dict, json_file, indent=4)

    print(f"Pesos salvos em 'chagas_weights.json'. Classe 1: {class_counts[1]}, Classe 0: {class_counts[0]}")

def get_train_test(df, split_column='Obito_MS', test_size=0.2):
    classe_1 = df[(df['Obito_MS'] == 1) & (df['Time'] < 5)]
    classe_0 = df[(df['Obito_MS'] == 0) | (df['Time'] == 5)]
    filtered_data = pd.concat([classe_1, classe_0])

    train, test = train_test_split(filtered_data, test_size=test_size, stratify=filtered_data[split_column], random_state=42)
    return train.reset_index(drop=True), test.reset_index(drop=True)

def compute_metrics(y_true, y_pred, threshold=0.5):
    y_pred_bin = np.where(y_pred > threshold, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (tp + tn) / (tp + fn + tn + fp)
    gmean = np.sqrt(recall * specificity)
    c_index = concordance_index(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'accuracy': acc,
        'geometric_mean': gmean,
        'c_index': c_index,
        'auc': auc
    }

