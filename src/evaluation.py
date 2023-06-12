from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from src.models import train_net

emotion2int = {'neutral':0, 'fear':1, 'disgust':2, 'happiness':3, 'boredom':4, 'sadness':5, 'anger':6}


################################################ Cross Validation ################################################


def CrossValidationByUser(X, y_true, model, users_id, model_name='model'):

    unique_users = np.unique(users_id)

    f1s = []
    precisions = []
    recalls = []

    for user in unique_users:
        indexes_val = np.where(users_id == user)[0]
        indexes_train = np.where(users_id!= user)[0]

        X_train = X[indexes_train]
        y_train = y_true[indexes_train]


        X_val = X[indexes_val]
        y_val = y_true[indexes_val]
            
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        
        f1s.append(f1_score(y_val, preds, average='macro', zero_division=0))
        precisions.append(precision_score(y_val, preds, average='macro', zero_division=0))
        recalls.append(recall_score(y_val, preds, average='macro', zero_division=0))

    print('|Model|F1|Precision|Recall|')
    print('|:----:|:---:|:---:|:----:|')
    
    means = [np.mean(metric) for metric in [f1s, precisions, recalls]]
    stds = [np.std(metric) for metric in [f1s, precisions, recalls]]

    print(f'|{model_name}|{means[0]:.02f}+-{stds[0]:.02f}|{means[1]:.02f}+-{stds[1]:.02f}|{means[2]:.02f}+-{stds[2]:.02f}|')

    return None


################################################ Final Metrics ################################################

def metrics_report(y_true, y_preds, mapping_label:Dict[str, int]=emotion2int) -> None:
    """
    Plots the confusion matrics and gives the F1, Precision, Recall for each metric
    """
    cm = confusion_matrix(y_true, y_preds)
    cm_df = pd.DataFrame(cm,
                     index = [label for label in mapping_label.keys()], 
                     columns = [label for label in mapping_label.keys()])
    plt.figure(figsize=(8,6))
    sns.set_style("whitegrid")
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()    
    accuracy = np.sum(np.diag(cm))/np.sum(cm)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    f1 = 2*recall*precision/(recall+precision)

    print('Results model with username where accuracy is: ', accuracy)
    print('|Emotion group|Precision |Recall|F1|')
    print('|:--:|:---:|:-:|:--:|')
    for k in range(recall.shape[0]):
        print(f'|{cm_df.index[k]}|{precision[k]:.02f}|{recall[k]:.02f}|{f1[k]:.02f}|')
    return None