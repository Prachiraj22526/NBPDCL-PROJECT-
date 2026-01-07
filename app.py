from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__, template_folder="templates")


# LOAD MODEL (.joblib)

MODEL_PATH = "stlf_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
    print("MODEL LOADED SUCCESSFULLY ✔")
except Exception as e:
    raise RuntimeError("Cannot load joblib model: " + str(e))

# FEATURES (ORDER MUST MATCH TRAINING)
FEATURES = [
    'hour', 'dayofweek', 'month', 'is_weekend',
    'lag_1', 'lag_2', 'lag_24', 'roll24_mean'
]


# HOME PAGE

@app.route("/")
def home():
    return render_template("index.html")


# API PREDICTION ROUTE

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Validate
    for feature in FEATURES:
        if feature not in data:
            return jsonify({"error": f"Missing feature: {feature}"}), 400

    try:
        # Convert to numpy array
        values = [float(data[f]) for f in FEATURES]
        arr = np.array(values).reshape(1, -1)

        # Predict
        prediction = model.predict(arr)[0]

        return jsonify({
            "prediction": float(prediction),
            "message": "Prediction successful!"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# RUN SERVER

if __name__ == "__main__":
    app.run(debug=True)
