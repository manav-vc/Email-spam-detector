# Email Spam Detector on AWS SageMaker
### COMP264 - Cloud Machine Learning - Group 7

This project trains a Multinomial Naive Bayes spam classifier on an email
dataset and deploys it as a real-time inference endpoint on Amazon
SageMaker. A simple HTML/JS frontend calls a public API Gateway URL that
is backed by an AWS Lambda function proxying requests to the SageMaker
endpoint.

```
Browser (index.html)
       |
       v
API Gateway (POST /predict)  ---->  AWS Lambda (lambda_function.py)
                                         |
                                         v
                              SageMaker Endpoint (inference.py)
                                         |
                                         v
                       Loaded Pipeline: CountVectorizer ->
                       TfidfTransformer -> MultinomialNB
```

---

## 1. Project structure

```
Group7project264_code/
├── README.md                       <- this file
├── train.py                        <- trains and saves the model
├── data/
│   └── Data5.csv                   <- dataset (502 labelled emails)
├── model/
│   ├── spam_pipeline.joblib        <- trained sklearn Pipeline
│   └── metrics.json                <- test accuracy / confusion matrix
├── sagemaker/
│   ├── inference.py                <- SageMaker entry point
│   ├── deploy.py                   <- packages + uploads + deploys
│   ├── invoke_test.py              <- smoke-test the endpoint
│   └── requirements.txt
├── lambda/
│   ├── lambda_function.py          <- Lambda -> SageMaker proxy
│   └── iam_policy.json             <- least-privilege IAM policy
└── frontend/
    └── index.html                  <- one-page UI for the demo
```

---

## 2. Local setup

```powershell
python -m venv .venv
.venv\Scripts\activate.bat
pip install scikit-learn pandas numpy joblib
```

If you also need to deploy to AWS, add `boto3`:
```powershell
pip install boto3
```

If you only need the SageMaker deploy script, make sure `boto3`
are installed in the same environment you use to run `deploy.py`.

## 3. Train the model

```powershell
python train.py --data-path data/Data5.csv --model-dir model
```

This produces:

* `model/spam_pipeline.joblib` - full sklearn Pipeline, ready to serve
* `model/metrics.json` - test accuracy, confusion matrix, best alpha

Test accuracy on the held-out 20% split: **~0.91**.

---

## 4. Deploy to AWS SageMaker

### 4.1 Prerequisites
* An AWS account
* A **SageMaker execution role** with `AmazonSageMakerFullAccess` and S3 access
* An S3 bucket (e.g. `group7-spam-detector-bucket`)
* `aws configure` completed locally OR an IAM user with access keys

### 4.2 Deploy
```powershell
cd sagemaker
python deploy.py --role-arn "arn:aws:iam::504133793968:role/SageMakerExecutionRole-SpamDetector" --bucket "group7-spam-detector-ca" --region "ca-central-1" --endpoint-name "spam-detector-endpoint" --instance-type "ml.t2.medium"
```

`deploy.py` will:
1. `tar.gz` the trained pipeline into `build/model.tar.gz`.
2. Upload it to `s3://YOUR_BUCKET/spam-detector/model.tar.gz`.
3. Create a `SKLearnModel` that references `inference.py`.
4. Deploy a real-time endpoint on the requested instance type.

### 4.3 Smoke-test the endpoint
```powershell
python invoke_test.py --endpoint spam-detector-endpoint --region ca-central-1
```

---

## 5. Expose the endpoint with Lambda + API Gateway

### 5.1 Create the Lambda function
1. AWS Console -> **Lambda** -> Create function -> Author from scratch.
2. Runtime: Python 3.11. Function name: `spam-detector-proxy`.
3. Upload `lambda/lambda_function.py` (or paste the code).
4. Add environment variables:
   * `ENDPOINT_NAME = spam-detector-endpoint`
   * `REGION = ca-central-1`
5. Attach the IAM policy in `lambda/iam_policy.json` so the function can
   call `sagemaker:InvokeEndpoint`.

### 5.2 Create the API Gateway trigger
1. Lambda -> Configuration -> Triggers -> Add trigger -> API Gateway.
2. Create a new **HTTP API**, security: Open.
3. Note the invoke URL, e.g.
   `https://abcd1234.execute-api.us-east-1.amazonaws.com/default/spam-detector-proxy`.
4. Enable **CORS** (the Lambda also returns the headers).

Test it:
```powershell
curl -X POST "https://abcd1234.execute-api.us-east-1.amazonaws.com/default/spam-detector-proxy" -H "Content-Type: application/json" -d '{"text":"Congratulations, you have won a FREE iPhone!"}'
```

---

## 6. Run the frontend

1. Open `frontend/index.html` in a browser (or host on S3 + CloudFront).
2. Paste the API Gateway URL into the *API URL* field.
3. Paste or pick a sample email and click **Check Spam**.

---

## 7. Cleaning up (to avoid charges)

```powershell
aws sagemaker delete-endpoint --endpoint-name spam-detector-endpoint
aws sagemaker delete-endpoint-config --endpoint-config-name spam-detector-endpoint
aws sagemaker delete-model --model-name spam-detector-endpoint
```
Also delete the Lambda function, API Gateway, and the S3 object if no
longer needed.

---

