#!/usr/bin/env python3
"""
Interactive Hand-on CLI Tester for the AWS Image Analysis Pipeline.
Run this script to test the pipeline with your own inputs interactively.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

# Setup environment
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["DYNAMODB_TABLE"] = "ImageAnalysisResults"
os.environ["SNS_TOPIC_ARN"] = "arn:aws:sns:eu-west-3:123456789012:security-alerts"

# Mock AWS SDK for local testing
sys.modules["boto3"] = MagicMock()
sys.modules["botocore"] = MagicMock()
sys.modules["botocore.exceptions"] = MagicMock()

import src.lambda_function as lambda_module


def print_step(num, title):
    print(f"\n\033[1;36m[{num}/5] {title}\033[0m")


def interactive_cli():
    print("\033[1;32m" + "=" * 65)
    print("  AWS SERVERLESS PIPELINE - INTERACTIVE HANDS-ON CONSOLE")
    print("=" * 65 + "\033[0m")
    print("Ce terminal interactif te donne la main pour tester le pipeline.\n")

    # 1. Ask user for image name
    default_name = "mon_image_vacances.jpg"
    user_input = input(f"👉 Entrez le nom de l'image à uploader [Défaut: {default_name}]: ").strip()
    image_name = user_input if user_input else default_name

    # 2. Ask user for category / scenario
    print("\n👉 Choisissez le type d'image pour simuler l'IA Rekognition :")
    print("   [1] 🐕 Animal de compagnie (Chien, Chat, Parc)")
    print("   [2] 🚗 Véhicule / Ville (Voiture, Route, Bâtiment)")
    print("   [3] 🏖️ Paysage (Plage, Soleil, Mer)")
    print("   [4] 🚨 Contenu sensible / Arme (Déclenchera une alerte SNS)")
    print("   [5] ✍️ Personnalisé (Saisir tes propres tags)")
    choice = input("Votre choix (1-5) [Défaut: 1]: ").strip() or "1"

    if choice == "1":
        labels = [{"Name": "Dog", "Confidence": 99.2}, {"Name": "Pet", "Confidence": 98.1}, {"Name": "Park", "Confidence": 85.0}]
        moderation = []
    elif choice == "2":
        labels = [{"Name": "Automobile", "Confidence": 97.5}, {"Name": "Street", "Confidence": 94.0}, {"Name": "Urban", "Confidence": 89.1}]
        moderation = []
    elif choice == "3":
        labels = [{"Name": "Beach", "Confidence": 99.0}, {"Name": "Ocean", "Confidence": 96.4}, {"Name": "Nature", "Confidence": 92.0}]
        moderation = []
    elif choice == "4":
        labels = [{"Name": "Weapon", "Confidence": 95.8}, {"Name": "Military", "Confidence": 88.0}]
        moderation = [{"Name": "Weapons & Violence", "ParentName": "Violence", "Confidence": 94.5}]
    else:
        raw_tags = input("Entrez des tags séparés par des virgules (ex: Chat, Maison, Jardin): ").strip()
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        labels = [{"Name": t, "Confidence": 95.0} for t in tags] or [{"Name": "CustomObject", "Confidence": 90.0}]
        is_mod = input("Ce contenu doit-il être considéré sensible pour déclencher une alerte ? (o/N): ").strip().lower()
        if is_mod == "o":
            moderation = [{"Name": "Flagged Content", "Confidence": 90.0}]
        else:
            moderation = []

    bucket_name = "image-analysis-pipeline-uploads-dev"

    # Execution simulation
    print_step("1", f"🪣 Amazon S3: Ingestion de l'image...")
    print(f"      Upload de '{image_name}' dans le bucket 's3://{bucket_name}/'")
    time.sleep(0.6)
    print("      \033[32m✔ Upload réussi -> Déclenchement de l'événement S3 ObjectCreated\033[0m")

    print_step("2", "⚡ AWS Lambda: Déclenchement automatique de la fonction...")
    s3_event = {
        "Records": [{
            "eventSource": "aws:s3",
            "eventName": "ObjectCreated:Put",
            "s3": {"bucket": {"name": bucket_name}, "object": {"key": image_name}}
        }]
    }
    time.sleep(0.6)
    print(f"      Handler invoqué: lambda_handler(event, context)")

    print_step("3", "🧠 Amazon Rekognition: Analyse d'image par Vision par Ordinateur...")
    lambda_module.rekognition_client.detect_labels.return_value = {"Labels": labels}
    lambda_module.rekognition_client.detect_moderation_labels.return_value = {"ModerationLabels": moderation}
    time.sleep(0.8)
    print("      Étiquettes (Labels) détectées :")
    for lbl in labels:
        print(f"        • \033[1m{lbl['Name']:<15}\033[0m Confiance : {lbl['Confidence']}%")

    if moderation:
        print("\n      \033[1;31m🚨 MODÉRATION : Contenu sensible détecté !\033[0m")
        for mod in moderation:
            print(f"        ⚠️  {mod['Name']} (Confiance: {mod['Confidence']}%)")
    else:
        print("      \033[32m✔ Modération : Aucun contenu sensible détecté.\033[0m")

    print_step("4", "📊 Amazon DynamoDB: Enregistrement dans la base NoSQL...")
    db_saved = {}
    lambda_module.dynamodb.Table.return_value.put_item.side_effect = lambda Item: db_saved.update(Item)
    time.sleep(0.6)

    sns_sent = []
    lambda_module.sns_client.publish.side_effect = lambda **kwargs: sns_sent.append(kwargs)

    # Call Lambda logic
    lambda_module.lambda_handler(s3_event, None)

    print(f"      Table : 'ImageAnalysisResults'")
    print(f"      Clé primaire (ImageName) : \033[1m{db_saved.get('ImageName')}\033[0m")
    print(f"      Horodatage (ProcessedAt) : {db_saved.get('ProcessedAt')}")
    print(f"      IsFlagged                : {db_saved.get('IsFlagged')}")

    print_step("5", "📬 Amazon SNS: Vérification des alertes...")
    time.sleep(0.5)
    if sns_sent:
        alert = sns_sent[0]
        print("\n\033[1;41;37m" + " " * 65)
        print("  🚨 EMAIL D'ALERTE ENVOYÉ VIA AMAZON SNS")
        print(" " * 65 + "\033[0m")
        print(f"  \033[1mSujet :\033[0m {alert['Subject']}")
        print(f"  \033[1mCorps du message :\033[0m\n{alert['Message']}\n")
    else:
        print("      \033[32m✔ Aucune alerte requise. Image conforme, 0 email envoyé.\033[0m")

    print("\n\033[1;32m" + "=" * 65)
    print("  ✔ TEST TERMINÉ AVEC SUCCÈS !")
    print("=" * 65 + "\033[0m\n")


if __name__ == "__main__":
    interactive_cli()
