# 🚀 Event-Driven Image Analysis Pipeline with AWS Rekognition & Terraform

![AWS](https://img.shields.io/badge/AWS-Serverless-orange?logo=amazon-aws)
![Terraform](https://img.shields.io/badge/IaC-Terraform-blueviolet?logo=terraform)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-brightgreen?logo=github-actions)

An automated, serverless, event-driven image analysis pipeline built on **Amazon Web Services (AWS)** and provisioned via **Infrastructure as Code (Terraform)**. 

When an image is uploaded to an **Amazon S3** bucket, an automated event triggers an **AWS Lambda** function. The function leverages **Amazon Rekognition** to detect objects, labels, and unsafe content, persists the metadata to **Amazon DynamoDB**, and dispatches immediate alerts through **Amazon SNS** if sensitive content is flagged.

---

## 🏛️ Architecture Overview

```
                      ┌─────────────────────────────────────────┐
                      │              AWS CLOUD                  │
                      │                                         │
┌──────────────┐      │   ┌──────────────┐     ┌────────────┐   │
│ User / Client├─────►│──►│  Amazon S3   ├────►│ AWS Lambda │   │
└──────────────┘Upload│   │ (Uploads)    │Event│  (Python)  │   │
                      │   └──────────────┘     └──────┬─────┘   │
                      │                               │         │
                      │           ┌───────────────────┼─────────┼───────────┐
                      │           │                   │         │           │
                      │           ▼                   ▼         ▼           │
                      │   ┌──────────────┐    ┌──────────────┐ ┌──────────┐ │
                      │   │Amazon        │    │Amazon        │ │Amazon    │ │
                      │   │Rekognition   │    │DynamoDB      │ │SNS       │ │
                      │   │(AI Detection)│    │(Metadata DB) │ │(Alerts)  │ │
                      │   └──────────────┘    └──────────────┘ └────┬─────┘ │
                      │                                             │       │
                      └─────────────────────────────────────────────┼───────┘
                                                                    ▼
                                                            ┌──────────────┐
                                                            │ Email Alert  │
                                                            └──────────────┘
```

---

## 🛠️ Key Technologies & AWS Services

- **Compute:** AWS Lambda (Python 3.12 / Boto3)
- **Storage:** Amazon S3 (Server-Side Encryption AES256, Block Public Access)
- **Database:** Amazon DynamoDB (NoSQL, On-Demand Capacity)
- **Machine Learning / AI:** Amazon Rekognition (Computer Vision: Object, Scene, and Content Moderation)
- **Alerting & Messaging:** Amazon SNS (Automated Email Notifications)
- **Monitoring & Security:** Amazon CloudWatch Logs & IAM (Principle of Least Privilege)
- **Infrastructure as Code (IaC):** HashiCorp Terraform
- **CI/CD:** GitHub Actions

---

## 📂 Repository Structure

```
├── .github/workflows/
│   └── ci.yml               # Automated GitHub Actions unit test workflow
├── architecture/            # Architecture diagrams
├── src/
│   └── lambda_function.py   # Core Lambda processing logic
├── tests/
│   ├── mock_event.json      # S3 test payload
│   └── test_lambda.py       # Python unit tests with mocks
├── terraform/
│   ├── main.tf              # Full infrastructure definition
│   ├── variables.tf         # Configurable parameters
│   ├── outputs.tf           # Resource ARNs and outputs
│   └── terraform.tfvars.example
├── .gitignore
└── README.md
```

---

## 🧪 Testing Locally

You can run the automated unit tests without an AWS account:

```bash
# Run unit tests
python3 -m unittest discover tests
```

---

## 🚀 Deployment (When ready to deploy to AWS)

1. **Configure AWS Credentials:**
   ```bash
   aws configure
   ```

2. **Deploy with Terraform:**
   ```bash
   cd terraform
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your email and preferred region

   terraform init
   terraform plan
   terraform apply
   ```

3. **Verify Deployment:**
   - Upload an image to the generated S3 bucket:
     ```bash
     aws s3 cp test.jpg s3://$(terraform output -raw s3_bucket_name)/
     ```
   - Check DynamoDB for the generated analysis record.
   - If flagged, check your inbox for an instant SNS security alert.

---

