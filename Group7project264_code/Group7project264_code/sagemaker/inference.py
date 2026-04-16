"""
SageMaker inference entry point for the Email Spam Detector.

SageMaker's SKLearn framework looks for four functions in this file
during online inference:
    model_fn(model_dir)               -> load the model artifact from disk
    input_fn(request_body, ctype)     -> deserialize HTTP request body
    predict_fn(input_object, model)   -> run prediction
    output_fn(prediction, accept)     -> serialize the response

The full sklearn Pipeline (CountVectorizer -> TfidfTransformer ->
MultinomialNB) is persisted with joblib as spam_pipeline.joblib.
"""

import os
import json
import joblib

MODEL_FILENAME = "spam_pipeline.joblib"
LABEL_MAP = {0: "ham", 1: "spam"}


def model_fn(model_dir):
    """Load the trained pipeline from the SageMaker model directory."""
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    return joblib.load(model_path)


def input_fn(request_body, request_content_type="application/json"):
    """Parse the incoming JSON payload.

    Accepted formats:
        {"text": "single email string"}
        {"texts": ["email 1", "email 2", ...]}
    """
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    payload = json.loads(request_body)
    if "texts" in payload:
        texts = payload["texts"]
    elif "text" in payload:
        texts = [payload["text"]]
    else:
        raise ValueError("Request JSON must contain 'text' or 'texts'.")

    # Normalise like training did
    texts = [str(t).lower().strip() for t in texts]
    return texts


def predict_fn(texts, model):
    """Return class predictions and probability for the 'spam' class."""
    preds = model.predict(texts).tolist()
    probs = model.predict_proba(texts).tolist()  # [[p_ham, p_spam], ...]

    results = []
    for text, pred, prob in zip(texts, preds, probs):
        results.append(
            {
                "input": text[:200],
                "prediction": LABEL_MAP[int(pred)],
                "label": int(pred),
                "spam_probability": round(float(prob[1]), 4),
                "ham_probability": round(float(prob[0]), 4),
            }
        )
    return results


def output_fn(prediction, accept="application/json"):
    """Serialize the list of prediction dicts as JSON."""
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps({"results": prediction}), accept
