terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# ------------------------------------------------------------------------------
# 1. Amazon S3 Bucket (Storage for Ingested Images)
# ------------------------------------------------------------------------------
resource "aws_s3_bucket" "image_bucket" {
  bucket        = "${var.project_name}-uploads-${var.environment}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "image_bucket_privacy" {
  bucket = aws_s3_bucket.image_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "image_bucket_crypto" {
  bucket = aws_s3_bucket.image_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ------------------------------------------------------------------------------
# 2. Amazon DynamoDB Table (Storage for Metadata and Analysis Results)
# ------------------------------------------------------------------------------
resource "aws_dynamodb_table" "analysis_results" {
  name         = "${var.project_name}-results-${var.environment}"
  billing_mode = "PAY_PER_REQUEST" # Free Tier friendly, 0 fixed cost
  hash_key     = "ImageName"
  range_key    = "ProcessedAt"

  attribute {
    name = "ImageName"
    type = "S"
  }

  attribute {
    name = "ProcessedAt"
    type = "S"
  }

  point_in_time_recovery {
    enabled = false
  }
}

# ------------------------------------------------------------------------------
# 3. Amazon SNS Topic (Alerting for Sensitive Content)
# ------------------------------------------------------------------------------
resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts-${var.environment}"
}

resource "aws_sns_topic_subscription" "email_sub" {
  count     = var.alert_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ------------------------------------------------------------------------------
# 4. IAM Role and Least-Privilege Policies for Lambda
# ------------------------------------------------------------------------------
resource "aws_iam_role" "lambda_exec_role" {
  name = "${var.project_name}-lambda-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action    = "sts:AssumeRole"
        Effect    = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_policy" "lambda_permissions" {
  name        = "${var.project_name}-lambda-policy-${var.environment}"
  description = "Least privilege policy for Image Analysis Lambda function"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # CloudWatch Logs
      {
        Effect   = "Allow"
        Action   = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      # S3 Read Object
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.image_bucket.arn}/*"
      },
      # DynamoDB Write
      {
        Effect   = "Allow"
        Action   = ["dynamodb:PutItem"]
        Resource = aws_dynamodb_table.analysis_results.arn
      },
      # Amazon Rekognition
      {
        Effect   = "Allow"
        Action   = [
          "rekognition:DetectLabels",
          "rekognition:DetectModerationLabels"
        ]
        Resource = "*"
      },
      # SNS Publish
      {
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = aws_sns_topic.alerts.arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_attach" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = aws_iam_policy.lambda_permissions.arn
}

# ------------------------------------------------------------------------------
# 5. AWS Lambda Function
# ------------------------------------------------------------------------------
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_file = "${path.module}/../src/lambda_function.py"
  output_path = "${path.module}/lambda_payload.zip"
}

resource "aws_lambda_function" "image_processor" {
  filename         = data.archive_file.lambda_zip.output_path
  function_name    = "${var.project_name}-processor-${var.environment}"
  role             = aws_iam_role.lambda_exec_role.arn
  handler          = "lambda_function.lambda_handler"
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  runtime          = "python3.12"
  timeout          = 30
  memory_size      = 256

  environment {
    variables = {
      DYNAMODB_TABLE       = aws_dynamodb_table.analysis_results.name
      SNS_TOPIC_ARN        = aws_sns_topic.alerts.arn
      CONFIDENCE_THRESHOLD = tostring(var.confidence_threshold)
    }
  }
}

# ------------------------------------------------------------------------------
# 6. S3 Event Trigger Wiring
# ------------------------------------------------------------------------------
resource "aws_lambda_permission" "allow_s3_invocation" {
  statement_id  = "AllowExecutionFromS3Bucket"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.image_processor.arn
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.image_bucket.arn
}

resource "aws_s3_bucket_notification" "bucket_notification" {
  bucket = aws_s3_bucket.image_bucket.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.image_processor.arn
    events              = ["s3:ObjectCreated:*"]
  }

  depends_on = [aws_lambda_permission.allow_s3_invocation]
}
