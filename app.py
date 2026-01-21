from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# ================= LOAD MODEL =================
with open("Heart_Disease.pkl", "rb") as f:
    model = pickle.load(f)

# If you saved scaler separately
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None


# ================= HOME PAGE =================
@app.route("/")
def home():
    return render_template("index.html")


# ================= PREDICTION =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        data = [
            float(request.form.get("age")),
            float(request.form.get("sex")),
            float(request.form.get("cp")),
            float(request.form.get("trestbps")),
            float(request.form.get("chol")),
            float(request.form.get("fbs")),
            float(request.form.get("restecg")),
            float(request.form.get("thalach")),
            float(request.form.get("exang")),
            float(request.form.get("oldpeak")),
            float(request.form.get("slope")),
            float(request.form.get("ca")),
            float(request.form.get("thal"))
        ]

        # Convert to numpy array
        input_data = np.array(data).reshape(1, -1)

        # Apply scaling if scaler exists
        if scaler:
            input_data = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_data)[0]

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})


# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)