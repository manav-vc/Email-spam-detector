"""
AWS Lambda handler that proxies requests from API Gateway
to the SageMaker Email Spam Detector endpoint.

Environment variables:
    ENDPOINT_NAME  - name of the deployed SageMaker endpoint
    REGION         - AWS region (default: us-east-1)

Expected API Gateway request body (JSON):
    {"text": "email content here"}

Response (JSON):
    {
        "prediction": "spam" | "ham",
        "spam_probability": 0.97,
        "ham_probability": 0.03
    }
"""

import json
import os
import base64
import boto3


ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "spam-detector-endpoint")
REGION = os.environ.get("REGION", "ca-central-1")

runtime = boto3.client("sagemaker-runtime", region_name=REGION)


CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
    "Content-Type": "application/json",
}


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps(body),
    }


def _http_method(event: dict) -> str:
    """Support both REST API (v1) and HTTP API (v2) event shapes."""
    return (
        event.get("httpMethod")
        or event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("requestContext", {}).get("httpMethod")
        or ""
    ).upper()


def _parse_body(event: dict) -> dict:
    body_raw = event.get("body") or "{}"
    if event.get("isBase64Encoded") and isinstance(body_raw, str):
        body_raw = base64.b64decode(body_raw).decode("utf-8")
    return json.loads(body_raw) if isinstance(body_raw, str) else body_raw


def lambda_handler(event, context):
    # CORS pre-flight
    if _http_method(event) == "OPTIONS":
        return _response(200, {"ok": True})

    try:
        body = _parse_body(event)

        text = body.get("text", "").strip()
        if not text:
            return _response(400, {"error": "Missing 'text' field in request body."})

        payload = json.dumps({"text": text})
        sm_response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=payload,
        )

        sm_body = json.loads(sm_response["Body"].read().decode("utf-8"))
        result = sm_body["results"][0]

        return _response(
            200,
            {
                "prediction": result["prediction"],
                "label": result["label"],
                "spam_probability": result["spam_probability"],
                "ham_probability": result["ham_probability"],
                "endpoint": ENDPOINT_NAME,
            },
        )

    except Exception as exc:
        # Log the full stack trace for CloudWatch
        import traceback
        traceback.print_exc()
        return _response(500, {"error": str(exc)})
