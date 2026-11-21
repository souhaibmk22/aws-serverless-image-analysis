import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Configure environment variables before importing lambda_function
os.environ["DYNAMODB_TABLE"] = "TestImageAnalysisResults"
os.environ["SNS_TOPIC_ARN"] = "arn:aws:sns:eu-west-3:123456789012:test-topic"
os.environ["CONFIDENCE_THRESHOLD"] = "70.0"

# If boto3/botocore is not installed in the local environment, mock them in sys.modules
if "boto3" not in sys.modules:
    mock_boto3 = MagicMock()
    mock_botocore = MagicMock()
    mock_exceptions = MagicMock()
    class MockClientError(Exception):
        pass
    mock_exceptions.ClientError = MockClientError
    mock_botocore.exceptions = mock_exceptions

    sys.modules["boto3"] = mock_boto3
    sys.modules["botocore"] = mock_botocore
    sys.modules["botocore.exceptions"] = mock_exceptions

from src.lambda_function import lambda_handler, send_sns_alert


class TestImageAnalysisLambda(unittest.TestCase):

    def setUp(self):
        with open("tests/mock_event.json") as f:
            self.sample_event = json.load(f)

    @patch("src.lambda_function.sns_client")
    @patch("src.lambda_function.dynamodb")
    @patch("src.lambda_function.rekognition_client")
    def test_lambda_successful_clean_image(self, mock_rekognition, mock_dynamodb, mock_sns):
        """Test processing a standard safe image without moderation flags."""
        # Mock Rekognition labels response
        mock_rekognition.detect_labels.return_value = {
            "Labels": [
                {"Name": "Dog", "Confidence": 98.45},
                {"Name": "Animal", "Confidence": 99.10},
                {"Name": "Pet", "Confidence": 95.00},
            ]
        }
        # Mock moderation response (no flags)
        mock_rekognition.detect_moderation_labels.return_value = {
            "ModerationLabels": []
        }

        # Mock DynamoDB Table put_item
        mock_table = MagicMock()
        mock_dynamodb.Table.return_value = mock_table

        response = lambda_handler(self.sample_event, None)

        # Assertions
        self.assertEqual(response["statusCode"], 200)
        body = json.loads(response["body"])
        self.assertEqual(len(body["processed"]), 1)
        self.assertEqual(body["processed"][0]["image"], "vacation photo.jpg")
        self.assertFalse(body["processed"][0]["flagged"])

        # Verify DynamoDB was called once
        mock_table.put_item.assert_called_once()
        inserted_item = mock_table.put_item.call_args[1]["Item"]
        self.assertEqual(inserted_item["ImageName"], "vacation photo.jpg")
        self.assertEqual(inserted_item["LabelsCount"], 3)
        self.assertFalse(inserted_item["IsFlagged"])

        # Verify SNS was NOT called
        mock_sns.publish.assert_not_called()

    @patch("src.lambda_function.sns_client")
    @patch("src.lambda_function.dynamodb")
    @patch("src.lambda_function.rekognition_client")
    def test_lambda_flagged_image_triggers_sns(self, mock_rekognition, mock_dynamodb, mock_sns):
        """Test processing an image flagged for moderation triggers an SNS alert."""
        mock_rekognition.detect_labels.return_value = {
            "Labels": [{"Name": "Graphic", "Confidence": 80.0}]
        }
        mock_rekognition.detect_moderation_labels.return_value = {
            "ModerationLabels": [
                {"Name": "Violence", "ParentName": "Explicit Content", "Confidence": 85.5}
            ]
        }

        mock_table = MagicMock()
        mock_dynamodb.Table.return_value = mock_table

        response = lambda_handler(self.sample_event, None)

        self.assertEqual(response["statusCode"], 200)
        body = json.loads(response["body"])
        self.assertTrue(body["processed"][0]["flagged"])

        # Verify DynamoDB has flagged flag set to True
        inserted_item = mock_table.put_item.call_args[1]["Item"]
        self.assertTrue(inserted_item["IsFlagged"])
        self.assertEqual(len(inserted_item["ModerationLabels"]), 1)

        # Verify SNS publish WAS called
        mock_sns.publish.assert_called_once()
        publish_kwargs = mock_sns.publish.call_args[1]
        self.assertIn("Violence", publish_kwargs["Message"])
        self.assertIn("vacation photo.jpg", publish_kwargs["Subject"])


if __name__ == "__main__":
    unittest.main()
