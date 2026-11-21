#!/usr/bin/env python3
"""
Zero-dependency Local Web Server for the AWS Image Analysis Pipeline.
Serves the interactive web interface and simulates S3, Lambda, Rekognition,
DynamoDB, and SNS locally.
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

os.makedirs(S3_DIR, exist_ok=True)
os.makedirs(WEB_DIR, exist_ok=True)

# Initialize local DB files if not present
if not os.path.exists(DYNAMODB_FILE):
    with open(DYNAMODB_FILE, "w") as f:
        json.dump([], f)

if not os.path.exists(SNS_FILE):
    with open(SNS_FILE, "w") as f:
        json.dump([], f)


def generate_labels_for_image(filename: str, file_bytes: bytes, hint: str = ""):
    """
    Simulates Amazon Rekognition computer vision analysis based on image content, metadata, and optional hints.
    """
    clean_text = f"{filename} {hint}".lower()
    words = re.findall(r"[a-zA-Z]+", clean_text)

    # Keywords mapping to realistic Rekognition label sets
    keyword_map = {
        "dog": [("Dog", 99.4), ("Pet", 98.2), ("Canine", 96.0), ("Mammal", 93.1), ("Animal", 98.9)],
        "chien": [("Dog", 99.4), ("Pet", 98.2), ("Canine", 96.0), ("Mammal", 93.1), ("Animal", 98.9)],
        "cat": [("Cat", 99.1), ("Pet", 98.0), ("Feline", 95.4), ("Kitten", 91.2), ("Animal", 98.5)],
        "chat": [("Cat", 99.1), ("Pet", 98.0), ("Feline", 95.4), ("Kitten", 91.2), ("Animal", 98.5)],
        "car": [("Car", 99.5), ("Automobile", 98.1), ("Vehicle", 97.4), ("Transportation", 95.0)],
        "voiture": [("Car", 99.5), ("Automobile", 98.1), ("Vehicle", 97.4), ("Transportation", 95.0)],
        "beach": [("Beach", 99.2), ("Coast", 97.8), ("Ocean", 96.5), ("Sea", 95.1), ("Nature", 94.0)],
        "plage": [("Beach", 99.2), ("Coast", 97.8), ("Ocean", 96.5), ("Sea", 95.1), ("Nature", 94.0)],
        "sunset": [("Sunset", 99.0), ("Sky", 98.4), ("Dusk", 95.1), ("Sunlight", 93.2)],
        "nature": [("Nature", 98.9), ("Outdoors", 97.5), ("Vegetation", 94.2), ("Landscape", 91.0)],
        "person": [("Person", 99.0), ("Human", 98.5), ("Face", 96.2), ("Portrait", 93.0)],
        "food": [("Food", 98.8), ("Meal", 97.1), ("Dish", 95.4), ("Cuisine", 92.0)],
        "weapon": [("Weapon", 96.5), ("Military", 91.2), ("Gun", 88.4)],
        "arme": [("Weapon", 96.5), ("Military", 91.2), ("Gun", 88.4)],
        "knife": [("Knife", 98.4), ("Weapon", 95.2), ("Blade", 94.1), ("Cutlery", 89.0)],
        "couteau": [("Knife", 98.4), ("Weapon", 95.2), ("Blade", 94.1), ("Cutlery", 89.0)],
        "blade": [("Blade", 97.0), ("Weapon", 94.5), ("Knife", 92.0)],
    }

    detected = []
    is_flagged = False
    moderation_labels = []

    # Check words in filename
    for word in words:
        if word in keyword_map:
            detected.extend(keyword_map[word])
            if word in ["weapon", "arme", "knife", "couteau", "blade"]:
                is_flagged = True
                moderation_labels.append({"Name": "Weapons & Violence (Knife)", "Confidence": 94.8})

    # Default labels if filename didn't match specific presets
    if not detected:
        # Detect basic file properties
        size_kb = len(file_bytes) / 1024
        detected = [
            ("Digital Image", 99.0),
            ("Photograph", 98.2),
            ("Visual Artwork", 93.5),
            ("Media Asset", 89.0),
        ]
        if size_kb > 500:
            detected.append(("High Resolution", 94.1))
        if words:
            detected.insert(0, (words[0].capitalize(), 91.5))

    # Format into Rekognition response style
    formatted_labels = [{"Name": name, "Confidence": round(float(conf), 1)} for name, conf in detected]
    return formatted_labels, is_flagged, moderation_labels


class PipelineHTTPHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.serve_file(os.path.join(WEB_DIR, "index.html"), "text/html")
        elif parsed.path == "/api/records":
            self.serve_dynamodb_records()
        elif parsed.path.startswith("/s3/"):
            # Serve uploaded images directly from storage/s3_bucket/
            file_name = os.path.basename(parsed.path)
            file_path = os.path.join(S3_DIR, file_name)
            if os.path.exists(file_path):
                mime, _ = mimetypes.guess_type(file_path)
                self.serve_file(file_path, mime or "application/octet-stream")
            else:
                self.send_error(404, "Object not found in S3 bucket")
        else:
            # Fallback to web directory
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
                base64_data = data.get("file_base64", "")

                # Decode file bytes
                if "," in base64_data:
                    base64_data = base64_data.split(",")[1]
                file_bytes = base64.b64decode(base64_data)

                # 1. Save to simulated S3 bucket
                s3_key = filename
                s3_path = os.path.join(S3_DIR, s3_key)
                with open(s3_path, "wb") as f:
                    f.write(file_bytes)

                # 2. Trigger simulated Lambda + Rekognition
                start_time = time.time()
                labels, is_flagged, moderation_labels = generate_labels_for_image(filename, file_bytes, custom_hint)
                latency_ms = round((time.time() - start_time) * 1000 + 45, 1)

                timestamp = datetime.now(timezone.utc).isoformat()

                # 3. Store in simulated DynamoDB
                db_record = {
                    "ImageName": s3_key,
                    "ProcessedAt": timestamp,
                    "Bucket": "image-analysis-pipeline-uploads-dev",
                    "FileSize": f"{round(len(file_bytes)/1024, 1)} KB",
                    "LabelsCount": len(labels),
                    "Labels": labels,
                    "IsFlagged": is_flagged,
                    "ModerationLabels": moderation_labels,
                    "ImageUrl": f"/s3/{s3_key}",
                }

                # Update persistent DynamoDB file
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

                # 4. Trigger simulated SNS Alert if flagged
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

                # Return response
                response_payload = {
                    "status": "SUCCESS",
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

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_payload).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ERROR", "message": str(e)}).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")

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
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(records).encode("utf-8"))


def run():
    server_address = ("127.0.0.1", PORT)
    httpd = HTTPServer(server_address, PipelineHTTPHandler)
    print("\n" + "=" * 65)
    print(f"🚀 AWS Pipeline Web Interface en ligne !")
    print(f"👉 Ouvre ton navigateur sur : http://localhost:{PORT}")
    print("=" * 65 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du serveur.")
        httpd.server_close()


if __name__ == "__main__":
    run()
