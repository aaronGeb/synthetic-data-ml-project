#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from catboost import CatBoostClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE


class CatBoostTrain:
    """Class to train a CatBoost model for binary classification"""

    def __init__(self, data_path: DataFrame, model_path="../models/catboost_model.pkl"):
        self.data_path = data_path
        self.model_path = model_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_data(self) -> DataFrame:
        """Load the data"""
        try:
            self.data = pd.read_csv(self.data_path)
            print("Data loaded successfully")
        except FileNotFoundError as e:
            print(f"Error loading the data: {e}")
            raise
        return self.data

    def split_data(self, data: DataFrame) -> DataFrame:
        """Split the data into train and test sets"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        # Split the dataset into train and test
        self.X_train, self.X_test = train_test_split(
            self.data, test_size=0.2, random_state=1
        )
        # Reset the index for both splits
        self.X_train = self.X_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        # Separate the target variable from the features
        self.y_train = self.X_train["suspected_fraud"].values
        self.y_test = self.X_test["suspected_fraud"].values
        # Remove the target variable from the features
        self.X_train = self.X_train.drop(columns=["suspected_fraud"])
        self.X_test = self.X_test.drop(columns=["suspected_fraud"])

    def upsample_data(self):
        """Upsample the data using SMOTE"""
        smote = SMOTE(random_state=1)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

    def train_model(self) -> None:
        """Train the model"""
        # Instantiate the model
        self.model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function="Logloss",
            verbose=100,
            random_state=1,
            eval_metric="AUC",
        )
        # Train the model
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=(self.X_test, self.y_test),
            early_stopping_rounds=500,
        )

    def evaluate_model(self) -> None:
        """Evaluate the model"""
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        # Calculate evaluation metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred)
        # Print the evaluation metrics
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"ROC AUC Score: {roc_auc}")
        # Print the confusion matrix
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

    def save_model(self) -> None:
        """Save the trained model to a file"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as file:
            pickle.dump(self.model, file)
        print(f"Model saved successfully")

    def run(self):
        """Load data, train model, evaluate model, and save model"""
        data = self.load_data()
        self.split_data(data)
        self.upsample_data()
        self.train_model()
        self.evaluate_model()
        self.save_model()


if __name__ == "__main__":
    catboost_train = CatBoostTrain(
        data_path="../data/labeled_data/fraud_encoded_labeled.csv"
    )
    catboost_train.run()
