import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier


def read_raw_data(path):
    """Reads data from a csv file
    """
    return pd.read_csv(path)


def data_preprocessing(data):
    """
    Type transformation
    Target encoding
    :param data: pandas dataframe with data
    :return: preprocessed data
    """

    # label encoding of target variable
    data['diagnosis'] = data['diagnosis'].replace(('M', 'B'), (1, 0))
    
    return data


def split_data_train_test(data, test_size=0.2):
    """
    Creates a train/test split of the dataset
    :param data: pandas dataframe
    :param test_size: proportion of test set
    :return: train/test dataframes
    """
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train,
                activation = "logistic",
                hidden_layer_sizes = (100, 50),
                learning_rate = "constant",
                random_state=49,
                max_iter=400):

    # Create model
    model = MLPClassifier(activation=activation,
                          hidden_layer_sizes=hidden_layer_sizes,
                          learning_rate=learning_rate,
                          random_state=random_state,
                          max_iter=max_iter)

    model.fit(X_train, y_train)

    return model


def eval_metrics(actual, pred):
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    roc_auc = roc_auc_score(actual, pred)
    return precision, recall, roc_auc


def mlflow_model(data):
    with mlflow.start_run():
        data = data_preprocessing(data)
        X_train, X_test, y_train, y_test = split_data_train_test(data)
        model = train_model(X_train, y_train)
        predicted = model.predict(X_test)
        (precision, recall, roc_auc) = eval_metrics(y_test, predicted)

        mlflow.log_params({})

        mlflow.log_metrics({"precision": precision, "recall": recall, "roc_auc": roc_auc})
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        mlflow.sklearn.log_model(model, "model")
