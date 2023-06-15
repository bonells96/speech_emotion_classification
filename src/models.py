from cProfile import label
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import time

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


int2emotion = {0:'neutral', 1:'fear', 2:'disgust', 3:'happiness', 4:'boredom', 5:'sadness', 6:'anger'}
code2emotion = {'W': 'anger', 'L':'boredom', 'E':'disgust', 'A':'fear',
                'F':'happiness', 'T':'sadness', 'N':'neutral'}

emotion2code = {emo:code for code, emo in code2emotion.items()}


################################################ Simple neural net ################################################

class StandardNet(nn.Module):
    def __init__(self, input_size, hidden_size=100):
        super(StandardNet, self).__init__()

        self.softmax = nn.Softmax(dim=1)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 7)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = (self.fc3(x))
        return out
    
################################################ Neural network with Batch Norm ################################################

class StandardNetBn(nn.Module):
    def __init__(self, input_size, hidden_size=100):
        super(StandardNetBn, self).__init__()

        self.softmax = nn.Softmax(dim=1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.ReLU()
            
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.ReLU()
            
        )
        self.fc3 = nn.Linear(hidden_size, 7)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out

    def predict(self, x):

        outputs = self.forward(x)
        preds = torch.argmax(outputs, dim=1)

        return preds

    def predict_sample(self, x, return_string=True):
        x = x.unsqueeze(0)
        
        outputs = self.forward(x)
        pred_class = torch.argmax(outputs, dim=1)
        emotion = int2emotion[pred_class.item()]
        if return_string:
            return {'label':pred_class.item(), 'emotion':emotion, 'code': emotion2code[emotion]}
        else:
            return pred_class.item()

    


################################################ Neural net Batch Norm and Dropout ################################################

class StandardNetBnD(nn.Module):
    def __init__(self, input_size, hidden_size=100):
        super(StandardNetBnD, self).__init__()

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.ReLU()
            
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.ReLU(),
            
        )
        self.fc3 = nn.Linear(hidden_size, 7)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        out = self.fc3(x)
        return out

    def predict_sample(self, x, return_string=True):
        x = x.unsqueeze(0)
        
        outputs = self.forward(x)
        pred_class = torch.argmax(outputs, dim=1)
        print(pred_class)
        if return_string:
            return int2emotion[pred_class.item()]
        else:
            return pred_class.item()





################################################ Cross Validation ################################################


def CrossValidationByUserDl(X, y_true, model, users_id, model_name='model', num_epochs=1000):
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
        model = train_net(model, X_train, y_train, num_epochs=num_epochs, verbose=False, plot_=False)
        end_train = time.time()

        start_inference = time.time()
        outputs = model(torch.Tensor(X_val))
        end_inference = time.time()

        preds = torch.argmax(outputs, dim=1).detach().numpy()
        
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


################################################ Training ################################################


def train_net(model, X, labels, num_epochs=10000, batch_size=25, learning_rate=1e-4, verbose=True, plot_=True):

    accsCat = {'neutral': [], 'fear':[], 'disgust': [], 'happiness': [], 'boredom': [], 'sadness': [], 'anger': [] }
    epochs = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = torch.utils.data.TensorDataset(torch.Tensor(X), (torch.LongTensor(labels)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        acc_cat = accuracy_per_category(torch.LongTensor(labels),model(torch.Tensor(X)) )
        for cat, acc in acc_cat.items():
            accsCat[cat].append(acc)

        epochs.append(epoch)
        if verbose:

            if (epoch+1)%100==0:
                print(f"Epoch {epoch + 1}: Loss = {running_loss}")
                print(f"Accuracy: {((len(X)-nb_errors(torch.LongTensor(labels), model(torch.Tensor(X))))/len(X))*100:.02f} % ")

    if plot_:

        fig, ax = plt.subplots(figsize=(12, 8))
        for cat, accs in accsCat.items():
            sns.set()
            sns.lineplot(x=epochs, y=accs, ax=ax, label = cat)

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Category over Epochs')

        plt.legend(title='Category')

    return model


    ################################################ Training and Validation ################################################



def train_net_with_val_results(model, X_train, y_train, X_test, y_test, num_epochs=1000, batch_size = 25,learning_rate=1e-4):
    acc_train = []
    acc_test = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train), (torch.LongTensor(y_train)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        

        preds_train = torch.argmax(model(torch.Tensor(X_train)), dim=1).detach().numpy()
        preds_test = torch.argmax(model(torch.Tensor(X_test)), dim=1).detach().numpy()

        acc_train.append(accuracy_score(y_train, preds_train))
        acc_test.append(accuracy_score(y_test, preds_test))
    return model, acc_train, acc_test

################################################ Helper Functions ################################################

def nb_errors(y_true, outputs):
    argmax_outputs = torch.argmax(outputs, dim=1)
    return torch.sum(torch.ne(argmax_outputs, y_true))


def accuracy_per_category(y_true, outputs, mapping_dict=int2emotion):
    # Convert predictions to class labels
    predicted_classes = torch.argmax(outputs, dim=1)
    
    # Calculate accuracy for each class
    class_accuracy = {}
    for class_label in range(outputs.size(1)):
        class_mask = y_true == class_label
        class_predictions = predicted_classes[class_mask]
        class_labels = y_true[class_mask]
        class_accuracy[mapping_dict[class_label]] = torch.sum(class_predictions == class_labels).item() / len(class_labels)
    
    return class_accuracy


