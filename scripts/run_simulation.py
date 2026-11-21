#!/usr/bin/env python3
"""
Interactive End-to-End Simulation Runner for Image Analysis Pipeline.
Simulates S3 upload events, Lambda invocation, Rekognition AI detection,
DynamoDB persistence, and SNS alerting directly in your terminal.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

# Add parent dir to path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set mock env
os.environ["DYNAMODB_TABLE"] = "ImageAnalysisResults-Demo"
os.environ["SNS_TOPIC_ARN"] = "arn:aws:sns:eu-west-3:123456789012:image-alerts"
os.environ["CONFIDENCE_THRESHOLD"] = "75.0"

# Mock boto3 for terminal simulation
from unittest.mock import MagicMock
mock_boto3 = MagicMock()
mock_botocore = MagicMock()
sys.modules["boto3"] = mock_boto3
sys.modules["botocore"] = mock_botocore
sys.modules["botocore.exceptions"] = MagicMock()

import src.lambda_function as lambda_module


def print_banner(text, char="="):
    print("\n" + char * 60)
    print(f"  {text}")
    print(char * 60)


def simulate_pipeline(scenario: str):
    print_banner(f"🚀 RUNNING PIPELINE SIMULATION: {scenario.upper()}")

    if scenario == "clean":
        image_name = "golden_retriever_park.jpg"
        bucket_name = "image-analysis-pipeline-uploads-dev"
        labels = [
            {"Name": "Dog", "Confidence": 99.4},
            {"Name": "Golden Retriever", "Confidence": 98.7},
            {"Name": "Pet", "Confidence": 97.2},
            {"Name": "Grass", "Confidence": 88.5},
            {"Name": "Park", "Confidence": 82.1},
        ]
        moderation_labels = []
    else:
        image_name = "suspicious_weapon_post.jpg"
        bucket_name = "image-analysis-pipeline-uploads-dev"
        labels = [
            {"Name": "Weapon", "Confidence": 94.2},
            {"Name": "Handgun", "Confidence": 91.8},
            {"Name": "Military", "Confidence": 79.5},
        ]
        moderation_labels = [
            {"Name": "Weapons & Violence", "ParentName": "Violence", "Confidence": 92.4}
        ]

    # Step 1: User uploads image to S3
    print(f"\n[1/5] 🪣 Amazon S3: Ingesting image...")
    time.sleep(0.5)
    print(f"      File: '{image_name}'")
    print(f"      Target Bucket: 's3://{bucket_name}/'")
    print(f"      Status: ✅ Upload complete (Event: s3:ObjectCreated:Put)")

    # Step 2: Triggering Lambda
    print(f"\n[2/5] ⚡ AWS Lambda: Function invoked by S3 EventTrigger...")
    time.sleep(0.5)
    s3_event = {
        "Records": [
            {
                "eventSource": "aws:s3",
                "eventName": "ObjectCreated:Put",
                "s3": {
                    "bucket": {"name": bucket_name},
                    "object": {"key": image_name},
                },
            }
        ]
    }
    print(f"      Payload received: {json.dumps(s3_event['Records'][0]['s3'])}")

    # Step 3: Rekognition Analysis
    print(f"\n[3/5] 🧠 Amazon Rekognition: Running computer vision AI detection...")
    time.sleep(0.8)
    lambda_module.rekognition_client.detect_labels.return_value = {"Labels": labels}
    lambda_module.rekognition_client.detect_moderation_labels.return_value = {
        "ModerationLabels": moderation_labels
    }

    print("      Detected Labels:")
    for lbl in labels:
        print(f"        • {lbl['Name']:<18} Confidence: {lbl['Confidence']}%")

    if moderation_labels:
        print("\n      🚨 MODERATION WARNING:")
        for mod in moderation_labels:
            print(f"        ⚠️  {mod['Name']} (Confidence: {mod['Confidence']}%)")
    else:
        print("      ✅ Content Moderation: Clean (No sensitive content detected)")

    # Step 4: DynamoDB storage
    print(f"\n[4/5] 📊 Amazon DynamoDB: Storing analysis metadata in table...")
    time.sleep(0.5)
    saved_item = {}
    def mock_put(Item):
        nonlocal saved_item
        saved_item = Item
    lambda_module.dynamodb.Table.return_value.put_item.side_effect = mock_put

    # Step 5: SNS Alerting
    sns_alerts_sent = []
    def mock_publish(TopicArn, Subject, Message):
        sns_alerts_sent.append({"Subject": Subject, "Message": Message})
    lambda_module.sns_client.publish.side_effect = mock_publish

    # Execute Lambda logic
    lambda_module.lambda_handler(s3_event, None)

    print(f"      Table: 'ImageAnalysisResults'")
    print(f"      Saved Record ID: '{saved_item.get('ImageName')}'")
    print(f"      Processed At:    {saved_item.get('ProcessedAt')}")
    print(f"      Is Flagged:      {saved_item.get('IsFlagged')}")

    # Step 5 Alert output
    print(f"\n[5/5] 📬 Amazon SNS: Checking alert conditions...")
    time.sleep(0.4)
    if sns_alerts_sent:
        alert = sns_alerts_sent[0]
        print(f"\n  ================== 📧 EMAIL NOTIFICATION DISPATCHED ==================")
        print(f"  To:       subscribed-admin@example.com")
        print(f"  Subject:  {alert['Subject']}")
        print(f"  Body:\n{alert['Message']}")
        print(f"  =====================================================================")
    else:
        print("      No alert needed. Image is safe. 0 emails sent.")

    print_banner("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "flagged":
        simulate_pipeline("flagged")
    else:
        simulate_pipeline("clean")
