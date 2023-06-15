from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from src.models import train_net
import time

emotion2int = {'neutral':0, 'fear':1, 'disgust':2, 'happiness':3, 'boredom':4, 'sadness':5, 'anger':6}


################################################ Cross Validation ################################################


def CrossValidationByUser(X, y_true, model, users_id, model_name='model'):
    """
    Perform cross-validation by user.

    Args:
        X (numpy.ndarray): Feature matrix.
        y_true (numpy.ndarray): True labels.
        model: Model to be evaluated.
        users_id (numpy.ndarray): User IDs corresponding to each sample.
        model_name (str, optional): Name of the model. Defaults to 'model'.

    """

    unique_users = np.unique(users_id)

    f1s = []
    precisions = []
    recalls = []
    train_times = []
    inference_times = []

    for user in unique_users:
        indexes_val = np.where(users_id == user)[0]
        indexes_train = np.where(users_id!= user)[0]

        X_train = X[indexes_train]
        y_train = y_true[indexes_train]


        X_val = X[indexes_val]
        y_val = y_true[indexes_val]
        
        start_train = time.time()
        model.fit(X_train, y_train)
        end_train = time.time()

        start_inference = time.time()
        preds = model.predict(X_val)
        end_inference = time.time()
        
        train_times.append(end_train-start_train)
        inference_times.append( X_val.shape[0]/(end_inference-start_inference) )
        f1s.append(f1_score(y_val, preds, average='macro', zero_division=0))
        precisions.append(precision_score(y_val, preds, average='macro', zero_division=0))
        recalls.append(recall_score(y_val, preds, average='macro', zero_division=0))

    print('|Model|F1|Precision|Recall|Train Time (s)|Inference Time (docs/s)|')
    print('|:----:|:---:|:---:|:----:|:----:|:---:|')
    
    means = [np.mean(metric) for metric in [f1s, precisions, recalls, train_times, inference_times]]
    stds = [np.std(metric) for metric in [f1s, precisions, recalls, train_times, inference_times]]

    output = f'|{model_name}|'
    for k in range(len(means)):
        output += f'{means[k]:.02f}+-{stds[k]:.02f}|'

    print(output)

    #print(f'|{model_name}|{means[0]:.02f}+-{stds[0]:.02f}|{means[1]:.02f}+-{stds[1]:.02f}|{means[2]:.02f}+-{stds[2]:.02f}|')

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
    print('|Emotion group|F1 |Precision|Recall|')
    print('|:--:|:---:|:-:|:--:|')
    for k in range(recall.shape[0]):
        print(f'|{cm_df.index[k]}|{f1[k]:.02f}|{precision[k]:.02f}|{recall[k]:.02f}|')
    return None