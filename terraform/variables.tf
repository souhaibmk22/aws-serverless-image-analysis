variable "aws_region" {
  description = "The AWS Region to deploy resources into"
  type        = "string"
  default     = "eu-west-3"
}

variable "project_name" {
  description = "Root prefix for naming project resources"
  type        = "string"
  default     = "image-analysis-pipeline"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = "string"
  default     = "dev"
}

variable "alert_email" {
  description = "Email address to receive moderation alerts via SNS"
  type        = "string"
  default     = ""
}

variable "confidence_threshold" {
  description = "Minimum confidence score for Rekognition label detection"
  type        = number
  default     = 75.0
}

variable "engine" {
  description = "Safety detection engine (CLIP_SUPERVISED, REKOGNITION)"
  type        = string
  default     = "CLIP_SUPERVISED"
}
