output "api_url" {
  value       = "https://${aws_lb.main.dns_name}"
  description = "Point the frontend's NEXT_PUBLIC_API_URL at this."
}

output "ecr_api_repository" {
  value = aws_ecr_repository.api.repository_url
}

output "ecr_worker_repository" {
  value = aws_ecr_repository.worker.repository_url
}

output "artifacts_bucket" {
  value       = aws_s3_bucket.artifacts.bucket
  description = "Model checkpoints and datasets. Never uploaded source code."
}

output "database_endpoint" {
  value     = aws_db_instance.postgres.endpoint
  sensitive = true
}
