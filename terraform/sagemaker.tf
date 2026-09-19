resource "aws_sagemaker_model" "predictops" {
  name               = "${var.project_name}-rul-model"
  execution_role_arn = aws_iam_role.sagemaker_execution_role.arn

  primary_container {
    image = var.docker_image_uri
  }

  tags = {
    Project = "PredictOps"
  }
}

resource "aws_sagemaker_endpoint_configuration" "predictops" {
  name = "${var.project_name}-rul-endpoint-config-${substr(md5(var.docker_image_uri), 0, 8)}"

  production_variants {
    variant_name = "PredictOpsVariant"
    model_name   = aws_sagemaker_model.predictops.name

    serverless_config {
      max_concurrency   = 1
      memory_size_in_mb = 2048
    }
  }

  tags = {
    Project = "PredictOps"
  }
}

resource "aws_sagemaker_endpoint" "predictops" {
  name                 = var.sagemaker_endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.predictops.name

  tags = {
    Project = "PredictOps"
  }
}   