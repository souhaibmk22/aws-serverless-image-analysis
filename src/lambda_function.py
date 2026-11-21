import json
import logging
import os
import urllib.parse
from datetime import datetime, timezone
import boto3
from botocore.exceptions import ClientError

# Setup structured logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables (configured via Terraform / Lambda env)
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "ImageAnalysisResults")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "75.0"))

# Initialize AWS clients
s3_client = boto3.client("s3")
rekognition_client = boto3.client("rekognition")
dynamodb = boto3.resource("dynamodb")
sns_client = boto3.client("sns")


def lambda_handler(event, context):
    """
    AWS Lambda entry point triggered by Amazon S3 ObjectCreated events.
    Analyzes uploaded images with Amazon Rekognition, saves metadata to DynamoDB,
    and publishes an alert to SNS if sensitive content is detected.
    """
    logger.info(f"Received S3 event: {json.dumps(event)}")
    processed_records = []

    for record in event.get("Records", []):
        try:
            # 1. Extract Bucket and Object Key from S3 Event
            bucket_name = record["s3"]["bucket"]["name"]
            raw_key = record["s3"]["object"]["key"]
            object_key = urllib.parse.unquote_plus(raw_key)

            logger.info(f"Processing object '{object_key}' from bucket '{bucket_name}'")

            # 2. Call Amazon Rekognition to detect labels (objects, concepts)
            labels_response = rekognition_client.detect_labels(
                Image={"S3Object": {"Bucket": bucket_name, "Name": object_key}},
                MaxLabels=10,
                MinConfidence=CONFIDENCE_THRESHOLD,
            )

            detected_labels = [
                {
                    "Name": label["Name"],
                    "Confidence": round(float(label["Confidence"]), 2),
                }
                for label in labels_response.get("Labels", [])
            ]

            # 3. Call Amazon Rekognition for Content Moderation
            moderation_response = rekognition_client.detect_moderation_labels(
                Image={"S3Object": {"Bucket": bucket_name, "Name": object_key}},
                MinConfidence=60.0,
            )

            moderation_labels = [
                {
                    "Name": mod["Name"],
                    "ParentName": mod.get("ParentName", ""),
                    "Confidence": round(float(mod["Confidence"]), 2),
                }
                for mod in moderation_response.get("ModerationLabels", [])
            ]

            is_flagged = len(moderation_labels) > 0
            timestamp = datetime.now(timezone.utc).isoformat()

            # 4. Save metadata to DynamoDB
            table = dynamodb.Table(DYNAMODB_TABLE_NAME)
            db_item = {
                "ImageName": object_key,
                "ProcessedAt": timestamp,
                "Bucket": bucket_name,
                "LabelsCount": len(detected_labels),
                "Labels": detected_labels,
                "IsFlagged": is_flagged,
                "ModerationLabels": moderation_labels,
            }

            table.put_item(Item=db_item)
            logger.info(f"Successfully recorded metadata for '{object_key}' in DynamoDB")

            # 5. Send SNS Alert if content was flagged
            if is_flagged and SNS_TOPIC_ARN:
                send_sns_alert(
                    image_name=object_key,
                    bucket=bucket_name,
                    moderation_labels=moderation_labels,
                )

            processed_records.append({"image": object_key, "status": "SUCCESS", "flagged": is_flagged})

        except ClientError as e:
            logger.error(f"AWS ClientError processing record: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing record: {str(e)}", exc_info=True)
            raise

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Processing completed successfully",
            "processed": processed_records,
        }),
    }


def send_sns_alert(image_name: str, bucket: str, moderation_labels: list):
    """Publishes a security notification to the Amazon SNS topic."""
    subject = f"⚠️ [ALERT] Sensitive Content Detected: {image_name}"
    labels_summary = "\n".join(
        [f"- {lbl['Name']} (Confidence: {lbl['Confidence']}%)" for lbl in moderation_labels]
    )

    message = (
        f"Automated Moderation Warning\n"
        f"-----------------------------------------\n"
        f"Image: {image_name}\n"
        f"Bucket: {bucket}\n"
        f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"Detected Moderation Labels:\n"
        f"{labels_summary}\n\n"
        f"Action: Please review this object in Amazon S3."
    )

    sns_client.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=subject,
        Message=message,
    )
    logger.info(f"Sent SNS alert for flagged image '{image_name}' to topic {SNS_TOPIC_ARN}")
