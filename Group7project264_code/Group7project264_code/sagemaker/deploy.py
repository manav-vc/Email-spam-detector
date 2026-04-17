"""Deploy the trained Email Spam Detector to Amazon SageMaker via boto3.

Pipeline:
    1. Package the trained pipeline + inference code into model.tar.gz.
    2. Upload model.tar.gz to S3.
    3. Create/Update SageMaker model, endpoint config, and endpoint.
"""

import argparse
import os
import tarfile
import time

import boto3
from botocore.exceptions import ClientError


SKLEARN_IMAGE_TAG = "1.2-1-cpu-py3"

# Region -> AWS account that hosts the SageMaker Scikit-learn DLC repository.
SKLEARN_ECR_ACCOUNTS = {
    "us-east-1": "683313688378",
    "us-east-2": "257758044811",
    "us-west-1": "746614075791",
    "us-west-2": "246618743249",
    "ca-central-1": "341280168497",
    "eu-west-1": "141502667606",
    "eu-west-2": "764974769150",
    "eu-west-3": "659782779980",
    "eu-central-1": "492215442770",
    "eu-north-1": "136758871317",
    "ap-southeast-1": "121021644041",
    "ap-southeast-2": "783357654285",
    "ap-northeast-1": "354813040037",
    "ap-northeast-2": "462105765813",
    "ap-south-1": "720646828776",
    "sa-east-1": "855470959533",
}


def sklearn_image_uri(region: str) -> str:
    account = SKLEARN_ECR_ACCOUNTS.get(region)
    if not account:
        raise ValueError(
            f"Unsupported region '{region}' for auto image resolution. "
            "Please add the ECR account mapping for this region."
        )
    return (
        f"{account}.dkr.ecr.{region}.amazonaws.com/"
        f"sagemaker-scikit-learn:{SKLEARN_IMAGE_TAG}"
    )


def package_model(model_path: str, output_tar: str, source_dir: str) -> str:
    """Tar.gz model artifact plus inference code for SageMaker."""
    os.makedirs(os.path.dirname(output_tar) or ".", exist_ok=True)

    inference_path = os.path.join(source_dir, "inference.py")
    requirements_path = os.path.join(source_dir, "requirements.txt")
    if not os.path.exists(inference_path):
        raise FileNotFoundError(f"Missing inference entry point: {inference_path}")

    with tarfile.open(output_tar, "w:gz") as tar:
        tar.add(model_path, arcname=os.path.basename(model_path))

        # SageMaker framework containers load user code from /opt/ml/model/code.
        tar.add(inference_path, arcname="code/inference.py")
        if os.path.exists(requirements_path):
            tar.add(requirements_path, arcname="code/requirements.txt")

    print(f"Packaged model -> {output_tar}")
    return output_tar


def upload_to_s3(tar_path: str, bucket: str, key: str, region: str) -> str:
    """Upload the tarball to S3 and return the s3:// URI."""
    s3 = boto3.client("s3", region_name=region)
    s3.upload_file(tar_path, bucket, key)
    uri = f"s3://{bucket}/{key}"
    print(f"Uploaded artifact -> {uri}")
    return uri


def deploy(model_uri: str, role: str, region: str, endpoint_name: str, instance: str) -> None:
    """Create/update SageMaker model and endpoint using boto3 only."""
    sm = boto3.client("sagemaker", region_name=region)
    image_uri = sklearn_image_uri(region)

    ts = int(time.time())
    model_name = f"{endpoint_name}-model-{ts}"
    endpoint_config_name = f"{endpoint_name}-config-{ts}"

    print(f"Creating model '{model_name}' ...")
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_uri,
            "Environment": {
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_REGION": region,
                "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
            },
        },
    )

    print(f"Creating endpoint config '{endpoint_config_name}' ...")
    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance,
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    endpoint_desc = None
    try:
        endpoint_desc = sm.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
    except ClientError as err:
        if "Could not find endpoint" in str(err):
            endpoint_exists = False
        else:
            raise

    if endpoint_exists:
        status = endpoint_desc.get("EndpointStatus", "Unknown")

        if status == "Failed":
            print(f"Endpoint '{endpoint_name}' is Failed. Deleting and recreating ...")
            sm.delete_endpoint(EndpointName=endpoint_name)

            print("Waiting for endpoint deletion ...")
            delete_waiter = sm.get_waiter("endpoint_deleted")
            delete_waiter.wait(EndpointName=endpoint_name)

            print(f"Creating endpoint '{endpoint_name}' ...")
            sm.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
            )
        else:
            print(f"Updating endpoint '{endpoint_name}' ...")
            sm.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
            )
    else:
        print(f"Creating endpoint '{endpoint_name}' ...")
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )

    print("Waiting for endpoint to become InService ...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)

    print(f"\nEndpoint deployed: {endpoint_name}")
    print("You can now invoke it via boto3 / the Lambda function.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="../model/spam_pipeline.joblib",
                        help="Local path to the trained joblib pipeline.")
    parser.add_argument("--role-arn", required=True,
                        help="SageMaker execution role ARN.")
    parser.add_argument("--bucket", required=True,
                        help="S3 bucket to upload the model tarball.")
    parser.add_argument("--prefix", default="spam-detector",
                        help="S3 key prefix.")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--endpoint-name", default="spam-detector-endpoint")
    parser.add_argument("--instance-type", default="ml.t2.medium")
    args = parser.parse_args()

    tar_path = "build/model.tar.gz"
    package_model(args.model_path, tar_path, source_dir=".")

    model_uri = upload_to_s3(
        tar_path=tar_path,
        bucket=args.bucket,
        key=f"{args.prefix}/model.tar.gz",
        region=args.region,
    )

    deploy(
        model_uri=model_uri,
        role=args.role_arn,
        region=args.region,
        endpoint_name=args.endpoint_name,
        instance=args.instance_type,
    )


if __name__ == "__main__":
    main()
