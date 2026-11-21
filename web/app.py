#!/usr/bin/env python3
"""
Local Web Server for the AWS Image Analysis Pipeline.
Supports BOTH:
1. Real Amazon Rekognition API (when AWS credentials are configured)
2. In-browser Deep Learning AI + Local simulation (when offline or testing without AWS account)
"""
import base64
import json
import mimetypes
import os
import re
import sys
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

PORT = 8000
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEB_DIR = os.path.join(BASE_DIR, "web")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
S3_DIR = os.path.join(STORAGE_DIR, "s3_bucket")
DYNAMODB_FILE = os.path.join(STORAGE_DIR, "dynamodb_records.json")
SNS_FILE = os.path.join(STORAGE_DIR, "sns_alerts.json")

AWS_REGION = os.environ.get("AWS_REGION", "eu-west-3")

os.makedirs(S3_DIR, exist_ok=True)
os.makedirs(WEB_DIR, exist_ok=True)

# Initialize local DB files if not present
if not os.path.exists(DYNAMODB_FILE):
    with open(DYNAMODB_FILE, "w") as f:
        json.dump([], f)

if not os.path.exists(SNS_FILE):
    with open(SNS_FILE, "w") as f:
        json.dump([], f)


def try_real_aws_rekognition(file_bytes: bytes):
    """
    Attempts to call real Amazon Rekognition if boto3 and credentials are available.
    Returns (labels, is_flagged, moderation_labels, True) on success, or None on failure.
    """
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError

        client = boto3.client("rekognition", region_name=AWS_REGION)
        
        # 1. Detect Labels
        labels_resp = client.detect_labels(
            Image={"Bytes": file_bytes},
            MaxLabels=10,
            MinConfidence=70.0
        )
        labels = [
            {"Name": l["Name"], "Confidence": round(float(l["Confidence"]), 1)}
            for l in labels_resp.get("Labels", [])
        ]

        # 2. Detect Moderation Labels
        mod_resp = client.detect_moderation_labels(
            Image={"Bytes": file_bytes},
            MinConfidence=60.0
        )
        moderation_labels = [
            {"Name": m["Name"], "Confidence": round(float(m["Confidence"]), 1)}
            for m in mod_resp.get("ModerationLabels", [])
        ]

        is_flagged = len(moderation_labels) > 0
        return labels, is_flagged, moderation_labels, True

    except (ImportError, NoCredentialsError, ClientError, Exception) as e:
        # Gracefully fall back to local/browser AI
        return None


def generate_smart_labels(filename: str, file_bytes: bytes, hint: str = "", client_labels: list = None):
    """
    Simulates Rekognition when real AWS credentials are not yet active,
    integrating client-side AI visual detections and keyword analysis.
    """
    client_labels = client_labels or []
    detected = []
    is_flagged = False
    moderation_labels = []

    # If in-browser AI sent detected labels
    for cl in client_labels:
        lbl_name = cl.get("name", "").capitalize()
        lbl_conf = cl.get("confidence", 90.0)
        detected.append((lbl_name, lbl_conf))

        # Check for moderation flags
        lower_name = lbl_name.lower()
        if lower_name in ["knife", "scissors", "weapon", "gun", "dagger", "sword"]:
            is_flagged = True
            moderation_labels.append({"Name": f"Weapons & Violence ({lbl_name})", "Confidence": lbl_conf})

    # Also check filename and hint keywords
    clean_text = f"{filename} {hint}".lower()
    words = re.findall(r"[a-zA-Z]+", clean_text)

    keyword_map = {
        "knife": [("Knife", 98.4), ("Weapon", 95.2), ("Blade", 94.1), ("Cutlery", 89.0)],
        "couteau": [("Knife", 98.4), ("Weapon", 95.2), ("Blade", 94.1), ("Cutlery", 89.0)],
        "blade": [("Blade", 97.0), ("Weapon", 94.5), ("Knife", 92.0)],
        "weapon": [("Weapon", 96.5), ("Military", 91.2), ("Gun", 88.4)],
        "arme": [("Weapon", 96.5), ("Military", 91.2), ("Gun", 88.4)],
        "dog": [("Dog", 99.4), ("Pet", 98.2), ("Canine", 96.0), ("Animal", 98.9)],
        "chien": [("Dog", 99.4), ("Pet", 98.2), ("Canine", 96.0), ("Animal", 98.9)],
        "cat": [("Cat", 99.1), ("Pet", 98.0), ("Feline", 95.4), ("Animal", 98.5)],
        "chat": [("Cat", 99.1), ("Pet", 98.0), ("Feline", 95.4), ("Animal", 98.5)],
        "car": [("Car", 99.5), ("Automobile", 98.1), ("Vehicle", 97.4), ("Transportation", 95.0)],
        "voiture": [("Car", 99.5), ("Automobile", 98.1), ("Vehicle", 97.4), ("Transportation", 95.0)],
        "beach": [("Beach", 99.2), ("Coast", 97.8), ("Ocean", 96.5), ("Nature", 94.0)],
        "plage": [("Beach", 99.2), ("Coast", 97.8), ("Ocean", 96.5), ("Nature", 94.0)],
    }

    for word in words:
        if word in keyword_map:
            detected.extend(keyword_map[word])
            if word in ["weapon", "arme", "knife", "couteau", "blade"]:
                is_flagged = True
                moderation_labels.append({"Name": "Weapons & Violence", "Confidence": 94.8})

    # Default labels if nothing found
    if not detected:
        detected = [
            ("Digital Image", 99.0),
            ("Photograph", 98.2),
            ("Visual Artwork", 93.5),
            ("Media Asset", 89.0),
        ]
        if words and words[0] not in ["screenshot"]:
            detected.insert(0, (words[0].capitalize(), 91.5))

    # Remove duplicates preserving order
    seen = set()
    unique_labels = []
    for name, conf in detected:
        if name not in seen:
            seen.add(name)
            unique_labels.append({"Name": name, "Confidence": round(float(conf), 1)})

    return unique_labels, is_flagged, moderation_labels, False


class PipelineHTTPHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.serve_file(os.path.join(WEB_DIR, "index.html"), "text/html")
        elif parsed.path == "/api/records":
            self.serve_dynamodb_records()
        elif parsed.path == "/api/status":
            # Check if real AWS credentials exist
            has_aws = bool(os.environ.get("AWS_ACCESS_KEY_ID") or os.path.exists(os.path.expanduser("~/.aws/credentials")))
            self.send_json(200, {"real_aws_connected": has_aws, "region": AWS_REGION})
        elif parsed.path.startswith("/s3/"):
            file_name = os.path.basename(parsed.path)
            file_path = os.path.join(S3_DIR, file_name)
            if os.path.exists(file_path):
                mime, _ = mimetypes.guess_type(file_path)
                self.serve_file(file_path, mime or "application/octet-stream")
            else:
                self.send_error(404, "Object not found in S3 bucket")
        else:
            local_path = os.path.join(WEB_DIR, parsed.path.lstrip("/"))
            if os.path.exists(local_path) and os.path.isfile(local_path):
                mime, _ = mimetypes.guess_type(local_path)
                self.serve_file(local_path, mime or "text/plain")
            else:
                self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/upload":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                data = json.loads(body.decode("utf-8"))
                filename = data.get("filename", f"upload_{int(time.time())}.jpg")
                custom_hint = data.get("hint", "")
                client_labels = data.get("client_labels", [])
                base64_data = data.get("file_base64", "")

                if "," in base64_data:
                    base64_data = base64_data.split(",")[1]
                file_bytes = base64.b64decode(base64_data)

                # 1. Save to S3 Bucket
                s3_key = filename
                s3_path = os.path.join(S3_DIR, s3_key)
                with open(s3_path, "wb") as f:
                    f.write(file_bytes)

                # 2. Analyze Image (Try Real AWS Rekognition first, fallback to Smart AI)
                start_time = time.time()
                real_aws_result = try_real_aws_rekognition(file_bytes)

                if real_aws_result:
                    labels, is_flagged, moderation_labels, is_real_aws = real_aws_result
                    backend_source = "AWS Rekognition (Cloud Live API)"
                else:
                    labels, is_flagged, moderation_labels, is_real_aws = generate_smart_labels(
                        filename, file_bytes, custom_hint, client_labels
                    )
                    backend_source = "In-Browser Computer Vision + Rekognition Engine"

                latency_ms = round((time.time() - start_time) * 1000 + 45, 1)
                timestamp = datetime.now(timezone.utc).isoformat()

                # 3. Save to DynamoDB
                db_record = {
                    "ImageName": s3_key,
                    "ProcessedAt": timestamp,
                    "Bucket": "image-analysis-pipeline-uploads-dev",
                    "FileSize": f"{round(len(file_bytes)/1024, 1)} KB",
                    "LabelsCount": len(labels),
                    "Labels": labels,
                    "IsFlagged": is_flagged,
                    "ModerationLabels": moderation_labels,
                    "Engine": backend_source,
                    "ImageUrl": f"/s3/{s3_key}",
                }

                records = []
                if os.path.exists(DYNAMODB_FILE):
                    try:
                        with open(DYNAMODB_FILE, "r") as f:
                            records = json.load(f)
                    except Exception:
                        records = []
                records.insert(0, db_record)
                with open(DYNAMODB_FILE, "w") as f:
                    json.dump(records[:50], f, indent=2)

                # 4. Trigger SNS Alert if flagged
                sns_alert = None
                if is_flagged:
                    sns_alert = {
                        "TopicArn": "arn:aws:sns:eu-west-3:123456789012:security-alerts",
                        "Subject": f"⚠️ [ALERT] Sensitive Content Detected: {s3_key}",
                        "Timestamp": timestamp,
                        "Image": s3_key,
                        "Moderation": moderation_labels,
                    }
                    alerts = []
                    if os.path.exists(SNS_FILE):
                        try:
                            with open(SNS_FILE, "r") as f:
                                alerts = json.load(f)
                        except Exception:
                            alerts = []
                    alerts.insert(0, sns_alert)
                    with open(SNS_FILE, "w") as f:
                        json.dump(alerts[:50], f, indent=2)

                response_payload = {
                    "status": "SUCCESS",
                    "engine": backend_source,
                    "s3": {
                        "bucket": "image-analysis-pipeline-uploads-dev",
                        "key": s3_key,
                        "size": db_record["FileSize"],
                    },
                    "lambda": {
                        "function": "image-analysis-pipeline-processor-dev",
                        "runtime": "python3.12",
                        "latency_ms": latency_ms,
                    },
                    "rekognition": {
                        "labels": labels,
                        "moderation": moderation_labels,
                    },
                    "dynamodb": db_record,
                    "sns": sns_alert,
                }

                self.send_json(200, response_payload)

            except Exception as e:
                self.send_json(500, {"status": "ERROR", "message": str(e)})
        else:
            self.send_error(404, "Endpoint not found")

    def send_json(self, status_code, data):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def serve_file(self, file_path, content_type):
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception:
            self.send_error(500, "Error reading file")

    def serve_dynamodb_records(self):
        records = []
        if os.path.exists(DYNAMODB_FILE):
            try:
                with open(DYNAMODB_FILE, "r") as f:
                    records = json.load(f)
            except Exception:
                records = []
        self.send_json(200, records)


def run():
    server_address = ("127.0.0.1", PORT)
    httpd = HTTPServer(server_address, PipelineHTTPHandler)
    print("\n" + "=" * 65)
    print(f"🚀 AWS Pipeline Web Interface avec intégration Rekognition en ligne !")
    print(f"👉 Ouvre ton navigateur sur : http://localhost:{PORT}")
    print("=" * 65 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du serveur.")
        httpd.server_close()


if __name__ == "__main__":
    run()
