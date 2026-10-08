# Local Kubernetes LLM platform on kind: cluster + kube-prometheus-stack + this repo's chart.
# Requires Docker (kind runs nodes as containers). Not applied in CI; CI runs fmt + validate only.

resource "kind_cluster" "this" {
  name           = var.cluster_name
  node_image     = var.node_image
  wait_for_ready = true

  kind_config {
    kind        = "Cluster"
    api_version = "kind.x-k8s.io/v1alpha4"

    node {
      role = "control-plane"

      # NodePorts exposed on the host: gateway 30080 -> 8080, Grafana 30300 -> 3000
      extra_port_mappings {
        container_port = 30080
        host_port      = 8080
      }
      extra_port_mappings {
        container_port = 30300
        host_port      = 3000
      }
    }
  }
}

provider "helm" {
  kubernetes = {
    host                   = kind_cluster.this.endpoint
    client_certificate     = kind_cluster.this.client_certificate
    client_key             = kind_cluster.this.client_key
    cluster_ca_certificate = kind_cluster.this.cluster_ca_certificate
  }
}

resource "helm_release" "monitoring" {
  count            = var.enable_monitoring ? 1 : 0
  name             = "monitoring"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  version          = "92.1.1"
  namespace        = "monitoring"
  create_namespace = true
  timeout          = 900

  values = [yamlencode({
    grafana = {
      service = { type = "NodePort", nodePort = 30300 }
      sidecar = {
        dashboards = { enabled = true, label = "grafana_dashboard", searchNamespace = "ALL" }
      }
    }
    prometheus = {
      prometheusSpec = {
        # pick up ServiceMonitors from any release, not only this chart's
        serviceMonitorSelectorNilUsesHelmValues = false
      }
    }
    alertmanager = { enabled = false }
  })]
}

resource "helm_release" "llm_platform" {
  name             = "llm-platform"
  chart            = "${path.module}/../helm/llm-platform"
  namespace        = var.namespace
  create_namespace = true
  timeout          = 1800 # first vLLM start downloads the model
  wait             = true

  values = [file("${path.module}/../helm/llm-platform/values-kind.yaml")]

  set = [
    { name = "gateway.image.tag", value = var.gateway_image_tag },
    { name = "vllm.model", value = var.model },
    { name = "vectorStore.kind", value = var.vector_store },
    { name = "qdrant.enabled", value = tostring(var.vector_store == "qdrant") },
    { name = "pgvector.enabled", value = tostring(var.vector_store == "pgvector") },
    { name = "monitoring.serviceMonitor.enabled", value = tostring(var.enable_monitoring) },
  ]

  depends_on = [helm_release.monitoring]
}
