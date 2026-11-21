output "s3_bucket_name" {
  description = "Name of the S3 Bucket configured for uploads"
  value       = aws_s3_bucket.image_bucket.id
}

output "dynamodb_table_name" {
  description = "Name of the DynamoDB Table holding results"
  value       = aws_dynamodb_table.analysis_results.id
}

output "lambda_function_name" {
  description = "Name of the deployed Lambda function"
  value       = aws_lambda_function.image_processor.function_name
}

output "sns_topic_arn" {
  description = "ARN of the SNS topic for alerts"
  value       = aws_sns_topic.alerts.arn
}
