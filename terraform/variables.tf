variable "aws_region" {
  description = "AWS region for PredictOps"
  type        = string
  default     = "ap-south-1"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "predictops"
}

variable "ecr_repository_name" {
  description = "ECR repository name"
  type        = string
  default     = "predictops-rul"
}

variable "sagemaker_endpoint_name" {
  description = "SageMaker endpoint name"
  type        = string
  default     = "predictops-rul-endpoint"
}

variable "docker_image_uri" {
  description = "Docker image URI from ECR"
  type        = string
  default     = ""
}