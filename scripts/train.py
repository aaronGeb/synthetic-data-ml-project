#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import pickle
import os
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class XGBoostTrain:
    """model for predicting the price of an items"""

    def __init__(self, data_path: DataFrame, model_path="../models/xgboost_model.pkl"):
        self.data_path = data_path
        self.model_path = model_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_data(self) -> DataFrame:
        """load the data"""
        self.data = pd.read_csv(self.data_path)
        return self.data

    def split_data(self, data: DataFrame) -> DataFrame:
        """split the data into train and test
        separate the target variable from the features"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        # split the datset into train and test
        self.X_train, self.X_test = train_test_split(
            self.data, test_size=0.2, random_state=1
        )
        # Reset the index for both splits
        self.X_train = self.X_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        # separate the target variable from the features
        self.y_train = self.X_train["log_price"].values
        self.y_test = self.X_test["log_price"].values

        # Remove the target variable from the features
        self.X_train = self.X_train.drop(columns=["log_price"])
        self.X_test = self.X_test.drop(columns=["log_price"])

    def train_model(self) -> None:
        """train the model"""
        # Instantiate the model
        sacler = StandardScaler()
        self.X_train = sacler.fit_transform(self.X_train)
        self.X_test = sacler.transform(self.X_test)
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            max_depth=9,
            min_child_weight=5,
            learning_rate=0.2,
            colsample_bytree=1.0,
            gamma=0.1,
            n_estimators=100,
            reg_alpha=0.1,
            reg_lambda=0,
            subsample=1.0,
            random_state=1,
        )
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully")

    def evaluate_model(self) -> None:
        """evaluate the model"""
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        # Evaluate the model
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f"Model has a R^2 score of {r2:.3f}")
        print(f"Model has a MSE score of {mse:.3f}")
        print(f"Model has a MAE score of {mae:.3f}")

    def save_model(self) -> None:
        """save the model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as file:
            pickle.dump(self.model, file)
        print(f"Model saved successfully")

    def run(self):
        """execute the training pipeline"""
        data = self.load_data()
        self.split_data(data)
        self.train_model()
        self.evaluate_model()
        self.save_model()


if __name__ == "__main__":
    trainer = XGBoostTrain(
        data_path="../data/labeled_data/synthetic_product_listings_gpt_4o_mini_encoded_labeled.csv"
    )
    trainer.run()
