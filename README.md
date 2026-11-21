# 🚀 Event-Driven Image Analysis & Unsupervised ML Pipeline with AWS & Terraform

![AWS](https://img.shields.io/badge/AWS-Serverless-orange?logo=amazon-aws)
![Terraform](https://img.shields.io/badge/IaC-Terraform-blueviolet?logo=terraform)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Unsupervised%20K--Means-blue)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-brightgreen?logo=github-actions)

An automated, serverless, event-driven image analysis and **Unsupervised Machine Learning anomaly detection pipeline** built on **Amazon Web Services (AWS)** and provisioned via **Infrastructure as Code (Terraform)**. 

The system combines managed cloud computer vision (**Amazon Rekognition**) with a custom **Unsupervised K-Means clustering algorithm** running serverless on **AWS Lambda** to detect suspicious content and anomalies without requiring pre-labeled training data.

---

## 🏛️ Architecture Overview

<div align="center">
  <img src="architecture/diagram.svg" alt="Architecture AWS & Machine Learning Non-Supervisé" width="100%">
</div>

<details>
<summary>🔍 Afficher le flux d'architecture détaillé (Mermaid)</summary>

```mermaid
flowchart LR
    Client["👤 Client / Web UI"] -->|"Upload Image"| S3["🪣 Amazon S3\n(Bucket Ingestion)"]
    S3 -->|"s3:ObjectCreated"| Lambda["⚡ AWS Lambda\n(Python 3.12 Orchestrator)"]
    
    Lambda -->|"Computer Vision"| Rekog["🧠 Amazon Rekognition\n(Managed AI)"]
    Lambda -->|"64-dim Feature Vector"| KMeans["📊 Modèle K-Means\n(Anomaly Detection)"]
    Lambda -->|"Store Metadata"| Dynamo["🗄️ Amazon DynamoDB\n(NoSQL Table)"]
    
    KMeans -->|"Si Distance > Seuil Critique"| SNS["📬 Amazon SNS\n(Topic Alerting)"]
    Rekog -->|"Si Modération Flagged"| SNS
    SNS -->|"Instant Notification"| Admin["🚨 Admin Security Alert\n(Email / SMS)"]
```
</details>

---

## 🧠 Unsupervised Machine Learning Engine (K-Means Anomaly Detection)

In high-security and content moderation environments, labeled threat datasets are rarely available. This project implements an **Unsupervised Learning** approach: the model learns the mathematical baseline of normal images and flags statistical outliers as anomalies.

### 1. Feature Engineering (`feature_extractor.py`)
Each incoming image is transformed into a **64-dimensional numerical feature vector** in $<5\text{ms}$:
- **Color Distributions (48 features):** 16-bin histograms across color projections.
- **Edge & Gradient Gradients (8 features):** Luminance contrast and edge density.
- **Statistical Moments (4 features):** Mean intensity, variance, standard deviation, and energy.
- **Compression & Highlight Density (4 features):** Shadow/highlight distribution and entropy proxies.

### 2. K-Means Clustering & Decision Boundary (`kmeans_engine.py`)
- Groups baseline images into $K=3$ distinct geometric clusters without supervision ($Y$ labels do not exist).
- Iteratively computes optimal cluster centroids $\mu_1, \mu_2, \dots, \mu_K$.
- Computes an automated statistical anomaly threshold:
  $$\tau = \mu_{\text{dist}} + 2.5 \times \sigma_{\text{dist}}$$

### 3. Real-Time Outlier Prediction
For any new image vector $x$, the Euclidean distance to the nearest centroid is evaluated:
$$d(x) = \min_{k} \|x - \mu_k\|_2$$
- If $d(x) \le \tau$ : The image conforms to normal patterns (**Normal**).
- If $d(x) > \tau$ : The image is a statistical outlier (**Suspicious / Anomaly Detected**).

### 4. Cloud Serverless Integration (`lambda_unsupervised.py`)
The trained model (`kmeans_model.json`) is loaded outside the Lambda handler (warm start) to deliver **sub-10ms serverless inference** directly inside the AWS ingestion pipeline.

---

## 🛠️ Key Technologies & AWS Services

- **Compute & Serverless ML:** AWS Lambda (Python 3.12 / Boto3 / NumPy-free pure ML)
- **Machine Learning:** Unsupervised K-Means Clustering, Euclidean Anomaly Detection, In-Browser Computer Vision
- **Storage:** Amazon S3 (Server-Side Encryption AES-256, Block Public Access)
- **Database:** Amazon DynamoDB (NoSQL, On-Demand Capacity, Single-digit ms latency)
- **Managed AI:** Amazon Rekognition (Computer Vision: Object Detection & Content Moderation)
- **Alerting & Messaging:** Amazon SNS (Publish/Subscribe Automated Security Alerts)
- **Monitoring & Security:** Amazon CloudWatch Logs & IAM (Principle of Least Privilege)
- **Infrastructure as Code (IaC):** HashiCorp Terraform
- **CI/CD:** GitHub Actions

---

## 📂 Repository Structure

```
├── .github/workflows/
│   └── ci.yml                     # Automated GitHub Actions test workflow
├── architecture/
│   └── diagram.svg                # Vector architecture diagram
├── ml_unsupervised/
│   ├── feature_extractor.py       # 64-dim image feature extractor
│   ├── kmeans_engine.py           # K-Means clustering & Euclidean outlier engine
│   ├── kmeans_model.json          # Exported centroids & decision threshold
│   ├── train_kmeans.py            # Unsupervised training script
│   ├── detect_anomaly.py          # CLI anomaly detector
│   └── lambda_unsupervised.py     # Production AWS Lambda ML handler
├── src/
│   └── lambda_function.py         # Rekognition & S3 event handler
├── tests/
│   ├── mock_event.json            # S3 test payload
│   └── test_lambda.py             # Automated unit tests
├── web/
│   ├── app.py                     # Local web server & AWS API bridge
│   └── index.html                 # Interactive drag-and-drop dashboard
├── terraform/
│   ├── main.tf                    # Complete infrastructure definition
│   ├── variables.tf               # Configurable parameters
│   ├── outputs.tf                 # Resource ARNs and outputs
│   └── terraform.tfvars.example
├── .gitignore
└── README.md
```

---

## 🧪 Testing Locally

### 1. Run Automated Unit Tests:
```bash
python3 -m unittest discover tests
```

### 2. Train and Run the Unsupervised ML Model:
```bash
# Train K-Means centroids and compute threshold
python3 ml_unsupervised/train_kmeans.py

# Test anomaly detection on any image file
python3 ml_unsupervised/detect_anomaly.py path/to/image.jpg
```

### 3. Launch the Interactive Web Dashboard:
```bash
python3 web/app.py
# Open http://localhost:8000 in your browser to test drag-and-drop uploads
```

---

## 🚀 Cloud Deployment (Terraform)

1. **Configure AWS Credentials:**
   ```bash
   aws configure
   ```

2. **Deploy with Terraform:**
   ```bash
   cd terraform
   cp terraform.tfvars.example terraform.tfvars
   # Configure alert_email and aws_region in terraform.tfvars

   terraform init
   terraform plan
   terraform apply
   ```

3. **Verify Deployment:**
   - Upload an image to the generated S3 bucket:
     ```bash
     aws s3 cp image.jpg s3://$(terraform output -raw s3_bucket_name)/
     ```
   - Check DynamoDB for the generated analysis record and ML anomaly score.
   - If an anomaly is detected, check your inbox for an instant SNS security alert.

4. **Cleanup (Zero cost):**
   ```bash
   terraform destroy
   ```
