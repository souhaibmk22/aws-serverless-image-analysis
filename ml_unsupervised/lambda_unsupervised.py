"""
AWS Lambda Function for Unsupervised Machine Learning Anomaly Detection.
Triggered by S3 ObjectCreated events.
Extracts 64-dimensional features, runs K-Means outlier prediction,
stores results in DynamoDB, and publishes security alerts to SNS.
"""
import json
import logging
import os
import urllib.parse
from datetime import datetime, timezone
import boto3
from botocore.exceptions import ClientError

from feature_extractor import extract_image_features
from kmeans_engine import KMeansAnomalyDetector

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "ImageAnalysisResults")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "kmeans_model.json")

# Initialize AWS clients
s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
sns_client = boto3.client("sns")

# Load model in memory outside handler (Warm Start optimization)
detector = KMeansAnomalyDetector()
if os.path.exists(MODEL_PATH):
    detector.load(MODEL_PATH)
    logger.info(f"Loaded K-Means model with threshold {detector.anomaly_threshold}")


def lambda_handler(event, context):
    """
    AWS Lambda entry point for Serverless Unsupervised Anomaly Detection.
    """
    logger.info(f"Received S3 event: {json.dumps(event)}")
    processed = []

    for record in event.get("Records", []):
        try:
            bucket_name = record["s3"]["bucket"]["name"]
            object_key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])

            # 1. Fetch Image Bytes from Amazon S3
            s3_response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
            image_bytes = s3_response["Body"].read()

            # 2. Extract 64-dim Feature Vector (Zero-dependency pure Python)
            features = extract_image_features(image_bytes)

            # 3. Predict Anomaly with K-Means
            prediction = detector.predict(features)
            is_anomaly = prediction["is_anomaly"]
            timestamp = datetime.now(timezone.utc).isoformat()

            # 4. Save metadata and Anomaly Metrics to DynamoDB
            table = dynamodb.Table(DYNAMODB_TABLE_NAME)
            item = {
                "ImageName": object_key,
                "ProcessedAt": timestamp,
                "Bucket": bucket_name,
                "ML_Model": "Unsupervised-KMeans-v1",
                "DistanceToCentroid": str(prediction["min_distance_to_centroid"]),
                "Threshold": str(prediction["threshold"]),
                "AnomalyScore": str(prediction["anomaly_score"]),
                "AssignedCluster": prediction["assigned_cluster"],
                "IsFlagged": is_anomaly,
                "Status": prediction["status"]
            }
            table.put_item(Item=item)
            logger.info(f"Recorded anomaly score for '{object_key}' in DynamoDB")

            # 5. Dispatch SNS Security Alert if Anomaly Detected
            if is_anomaly and SNS_TOPIC_ARN:
                subject = f"🚨 [ML ANOMALY] Suspicious Outlier Detected: {object_key}"
                message = (
                    f"AWS Unsupervised Machine Learning Alert\n"
                    f"-----------------------------------------\n"
                    f"Image: {object_key}\n"
                    f"Bucket: {bucket_name}\n"
                    f"Model: Unsupervised K-Means Outlier Detection\n"
                    f"Distance to Centroid: {prediction['min_distance_to_centroid']} (Threshold: {prediction['threshold']})\n"
                    f"Anomaly Score: {prediction['anomaly_score']} / 1.0\n"
                    f"Decision: Outlier / Threat detected\n\n"
                    f"Please review object in Amazon S3."
                )
                sns_client.publish(
                    TopicArn=SNS_TOPIC_ARN,
                    Subject=subject,
                    Message=message
                )
                logger.info(f"Dispatched SNS alert for outlier '{object_key}'")

            processed.append({"image": object_key, "is_anomaly": is_anomaly})

        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}", exc_info=True)
            raise

    return {
        "statusCode": 200,
        "body": json.dumps({"status": "SUCCESS", "processed": processed})
    }
