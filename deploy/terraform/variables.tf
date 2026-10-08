variable "cluster_name" {
  description = "kind cluster name"
  type        = string
  default     = "llm-platform"
}

variable "node_image" {
  description = "kindest/node image (Kubernetes version)"
  type        = string
  default     = "kindest/node:v1.33.1"
}

variable "namespace" {
  description = "Namespace for the platform release"
  type        = string
  default     = "llm"
}

variable "gateway_image_tag" {
  description = "Tag of the llm-platform-gateway image loaded into kind (make kind-load)"
  type        = string
  default     = "dev"
}

variable "model" {
  description = "Hugging Face model id served by vLLM"
  type        = string
  default     = "Qwen/Qwen2.5-0.5B-Instruct"
}

variable "enable_monitoring" {
  description = "Install kube-prometheus-stack (Prometheus Operator + Grafana with dashboard sidecar)"
  type        = bool
  default     = true
}

variable "vector_store" {
  description = "qdrant or pgvector"
  type        = string
  default     = "qdrant"
  validation {
    condition     = contains(["qdrant", "pgvector"], var.vector_store)
    error_message = "vector_store must be qdrant or pgvector."
  }
}
