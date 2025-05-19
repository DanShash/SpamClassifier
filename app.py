from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained pipeline
with open("spam_classifier.pkl", "rb") as f:
    pipeline = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get email text from JSON request
        data = request.get_json()
        if not data or "email" not in data:
            return jsonify({"error": "Missing 'email' field in JSON payload"}), 400
        
        email_text = data["email"]
        
        # Convert single email to array for pipeline
        X_new = np.array([email_text], dtype=object)
        
        # Predict using the pipeline
        prediction = pipeline.predict(X_new)[0]
        probability = pipeline.predict_proba(X_new)[0][1]  # Probability of spam
        
        # Map prediction to label
        label = "spam" if prediction == 1 else "ham"
        
        return jsonify({
            "prediction": label,
            "probability": round(float(probability), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)