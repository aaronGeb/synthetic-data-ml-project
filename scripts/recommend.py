#!/usr/bin/env python3
import pandas as pd
import pickle
from surprise import SVD


class RecommendationSystem:
    def __init__(
        self,
        model_path="../models/recommendation.pkl",
        data_path="../data/user_interactions.csv",
    ):
        """
        Load the trained recommendation model.
        """
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.df = pd.read_csv(self.data_path)

    def load_model(self):
        """Load a pre-trained model."""
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        print("Model loaded successfully.")

    def recommend_products(self, user_id, top_n=5):
        """
        Recommend top N products for a given user.
        :param user_id: ID of the user
        :param top_n: Number of recommendations to return
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")

        # Get all unique products
        all_products = self.df["product_id"].unique()

        # Get products already rated by the user
        rated_products = self.df[self.df["user_id"] == user_id]["product_id"].tolist()

        # Find products the user has NOT rated
        products_to_predict = [p for p in all_products if p not in rated_products]

        # Predict ratings for these products
        predictions = [self.model.predict(user_id, p) for p in products_to_predict]

        # Sort by highest predicted rating
        top_recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[
            :top_n
        ]

        return [(pred.iid, pred.est) for pred in top_recommendations]


if __name__ == "__main__":
    rec_sys = RecommendationSystem()
    rec_sys.load_model()
    user_id = 576
    recommendations = rec_sys.recommend_products(user_id)
    print("Recommended products:", recommendations)
