#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pandas import DataFrame
import pickle


class PricePredictor:
    """class for loading the model and making predictions"""

    def __init__(self, model_path: str):
        """
        Initializes the PricePredictor class.
        Args:
        model_path (str): Path to the trained model file (pickle format)
        """
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """load the model from the path"""
        try:
            with open(self.model_path, "rb") as file:
                self.model = pickle.load(file)
            print("Model loaded successfully")
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f"Error loading the model:{e}")
            raise

    def predict_price(self, input_data: DataFrame):
        """predict the price using the model"""
        if self.model is None:
            raise Exception("Model not loaded")
            return None
        try:
            prediction = self.model.predict(input_data)
            return prediction
        except Exception as e:
            print(f"Error predicting the price:{e}")
            raise


if __name__ == "__main__":
    predictor = PricePredictor(model_path="../models/xgboost_model.pkl")
    predictor.load_model()
    new_data = pd.DataFrame(
        {"category": [2], "brand": [8], "condition": [0], "seller_reputation": [4]}
    )
    prediction = predictor.predict_price(new_data)
    if prediction is not None:
        print(f"The predicted price is: {prediction}")
