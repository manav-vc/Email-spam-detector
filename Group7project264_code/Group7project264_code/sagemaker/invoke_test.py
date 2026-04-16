"""
Quick smoke-test for a deployed SageMaker endpoint.

Usage:
    python invoke_test.py --endpoint spam-detector-endpoint --region us-east-1
"""

import argparse
import json
import boto3


SAMPLES = [
    "Hi team, please find attached the notes from yesterday's meeting.",
    "Congratulations! You've won a FREE iPhone, click here to claim NOW!",
    "Your account has been compromised, verify your password immediately!",
    "Looking forward to catching up with you this weekend.",
    "URGENT: Send us your bank details to claim your $1,000,000 prize!",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    runtime = boto3.client("sagemaker-runtime", region_name=args.region)

    payload = json.dumps({"texts": SAMPLES})
    response = runtime.invoke_endpoint(
        EndpointName=args.endpoint,
        ContentType="application/json",
        Accept="application/json",
        Body=payload,
    )
    body = json.loads(response["Body"].read().decode("utf-8"))

    print("\nSageMaker endpoint response:")
    for item in body["results"]:
        tag = item["prediction"].upper()
        prob = item["spam_probability"]
        print(f"  [{tag:<4}] (p_spam={prob:.2f})  {item['input'][:70]}")


if __name__ == "__main__":
    main()
