"""
Fraud detector backend.
Loads the trained SMS and email models, and serves a simple web page
where you can paste a message and get a prediction back.
"""

import os
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

MODELS_DIR = "models"

print("Loading models...")
sms_vectorizer = joblib.load(os.path.join(MODELS_DIR, "sms_tfidf_vectorizer.joblib"))
sms_model = joblib.load(os.path.join(MODELS_DIR, "sms_logreg_model.joblib"))

email_vectorizer = joblib.load(os.path.join(MODELS_DIR, "email_tfidf_vectorizer.joblib"))
email_model = joblib.load(os.path.join(MODELS_DIR, "email_logreg_model.joblib"))
print("Models loaded.")

MODELS = {
    "sms": {"vectorizer": sms_vectorizer, "model": sms_model},
    "email": {"vectorizer": email_vectorizer, "model": email_model},
}


def classify(channel, text):
    """Run the right model for the channel and return a result dictionary."""
    bundle = MODELS[channel]
    vec = bundle["vectorizer"].transform([text])
    proba = bundle["model"].predict_proba(vec)[0]
    spam_prob = float(proba[1])  # probability the message is spam/phishing

    if spam_prob >= 0.75:
        tier = "block"
    elif spam_prob >= 0.4:
        tier = "review"
    else:
        tier = "safe"

    return {"channel": channel, "spam_probability": spam_prob, "tier": tier}


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    channel = "sms"
    text = ""

    if request.method == "POST":
        channel = request.form.get("channel", "sms")
        text = request.form.get("text", "")
        if channel in MODELS and text.strip():
            result = classify(channel, text)

    return render_template("index.html", result=result, channel=channel, text=text)

if __name__ == "__main__":
    app.run(debug=False, port=5001)