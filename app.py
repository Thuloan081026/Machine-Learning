import os, sys
from flask import Flask, request, jsonify, render_template

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from predict import load_model_artifact, predict_customer_segment

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
_artifact = None

def get_artifact():
    global _artifact
    if _artifact is None:
        save_path = os.path.join(BASE_DIR, "models", "save_model", "kmeans_model.json")
        _artifact = load_model_artifact(save_path)
    return _artifact

@app.route("/", methods=["GET"])
def index():
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    if os.path.exists(template_path):
        return render_template("index.html")
    return jsonify({
        "project": "Customer Segmentation API (K-Means Clustering From Scratch)",
        "algorithm": "100% NumPy / Pandas (Zero Scikit-Learn)",
        "endpoints": {
            "GET /": "API Information",
            "GET /health": "Health Check",
            "POST /predict": "Predict Customer Segment Persona"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    try:
        get_artifact()
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body phai la JSON"}), 400

    income = data.get("Annual_Income", data.get("income"))
    spending = data.get("Spending_Score", data.get("spending"))

    if income is None or spending is None:
        return jsonify({"error": "Thieu truong: Annual_Income va Spending_Score"}), 400

    try:
        artifact = get_artifact()
        res = predict_customer_segment(float(income), float(spending), artifact)
        return jsonify({
            "status": "success",
            "input": {"Annual_Income": float(income), "Spending_Score": float(spending)},
            "cluster": res["cluster"],
            "persona": res["persona"],
            "distance_to_centroid": round(res["distance_to_centroid"], 4)
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=False)
