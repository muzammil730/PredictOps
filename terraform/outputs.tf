output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.predictops.repository_url
}

output "sagemaker_endpoint_name" {
  description = "SageMaker endpoint name"
  value       = aws_sagemaker_endpoint.predictops.name
}

output "sagemaker_execution_role_arn" {
  description = "SageMaker execution role ARN"
  value       = aws_iam_role.sagemaker_execution_role.arn
}