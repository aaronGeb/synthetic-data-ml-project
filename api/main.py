from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained models
models = {
    "price_prediction": pickle.load(open("models/price_prediction.pkl", "rb")),
    "fraud_detection": pickle.load(open("models/fraud_detection.pkl", "rb")),
    "recommendation": pickle.load(open("models/recommendation.pkl", "rb")),
}


@app.route("/")
def home():
    return "Welcome to your shope!"


# Price Prediction Endpoint
@app.route("/predict_price", methods=["POST"])
def predict_price():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = models["price_prediction"].predict(features).tolist()
        return jsonify({"price_prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})


# Fraud Detection Endpoint
@app.route("/detect_fraud", methods=["POST"])
def detect_fraud():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = models["fraud_detection"].predict(features).tolist()
        return jsonify({"fraud_prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})


# Recommendation Endpoint
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        user_id = request.get_json().get("user_id")
        prediction = models["recommendation"].predict([[user_id]]).tolist()
        return jsonify({"recommendations": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
