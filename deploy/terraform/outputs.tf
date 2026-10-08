output "kubeconfig_path" {
  description = "kubeconfig for the kind cluster"
  value       = kind_cluster.this.kubeconfig_path
}

output "gateway_port_forward" {
  description = "Reach the gateway"
  value       = "kubectl -n ${var.namespace} port-forward svc/llm-platform-gateway 8080:80"
}

output "grafana_url" {
  description = "Grafana (admin password: kubectl -n monitoring get secret monitoring-grafana -o jsonpath='{.data.admin-password}' | base64 -d)"
  value       = var.enable_monitoring ? "http://localhost:3000" : null
}
