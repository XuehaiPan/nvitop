# nvitop-exporter Kubernetes DaemonSet

This directory contains Kubernetes manifests to deploy `nvitop-exporter` as a DaemonSet for monitoring GPU metrics across all nodes in your cluster.

## 📋 Prerequisites

- Kubernetes cluster with GPU nodes
- NVIDIA GPU drivers installed on nodes
- NVIDIA Container Toolkit (for GPU access)
- Prometheus Operator (optional, for ServiceMonitor)

## 🚀 Quick Deployment

### Option 1: Using Kustomize (Recommended)
```bash
kubectl apply -k k8s/
```

### Option 2: Manual Deployment
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/serviceaccount.yaml
kubectl apply -f k8s/daemonset.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/servicemonitor.yaml  # Only if using Prometheus Operator
```

## 📁 Files Overview

| File | Description |
|------|-------------|
| `namespace.yaml` | Creates the `monitoring` namespace |
| `serviceaccount.yaml` | ServiceAccount and RBAC permissions |
| `daemonset.yaml` | Main DaemonSet configuration |
| `service.yaml` | Headless service for metrics exposure |
| `servicemonitor.yaml` | Prometheus Operator ServiceMonitor |
| `kustomization.yaml` | Kustomize configuration |

## ⚙️ Configuration

### Node Selection

The DaemonSet uses a node selector to only deploy on GPU nodes:
```yaml
nodeSelector:
  accelerator: nvidia-gpu
```

**To label your GPU nodes:**
```bash
kubectl label nodes <node-name> accelerator=nvidia-gpu
```

### Resource Limits

Default resource configuration:
```yaml
resources:
  requests:
    memory: "64Mi"
    cpu: "50m"
  limits:
    memory: "128Mi"
    cpu: "200m"
```

### Command Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--bind-address` | `0.0.0.0` | Address to bind to |
| `--port` | `8080` | Port to listen on |
| `--interval` | `10` | Metrics collection interval (seconds) |

## 🔍 Monitoring & Observability

### Prometheus Integration

The DaemonSet includes annotations for automatic Prometheus discovery:
```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8080"
  prometheus.io/path: "/metrics"
```

### Health Checks

- **Liveness Probe**: HTTP GET `/metrics` every 30s
- **Readiness Probe**: HTTP GET `/metrics` every 10s

### Accessing Metrics

1. **Port Forward** (for testing):
   ```bash
   kubectl port-forward -n monitoring daemonset/nvitop-exporter 8080:8080
   curl http://localhost:8080/metrics
   ```

2. **Via Service**:
   ```bash
   kubectl get svc -n monitoring nvitop-exporter
   ```

## 🛠️ Customization

### Different Image Tag
Edit `kustomization.yaml`:
```yaml
images:
  - name: ghcr.io/ntheanh201/nvitop-exporter
    newTag: v1.0.0  # Change to desired tag
```

### Different Namespace
Edit `kustomization.yaml`:
```yaml
metadata:
  namespace: your-namespace
```

### Custom Node Selector
Edit `daemonset.yaml`:
```yaml
nodeSelector:
  your-label: your-value
```

### Additional Tolerations
Edit `daemonset.yaml`:
```yaml
tolerations:
- key: your-taint
  operator: Exists
  effect: NoSchedule
```

## 🔧 Troubleshooting

### Check DaemonSet Status
```bash
kubectl get daemonset -n monitoring nvitop-exporter
kubectl describe daemonset -n monitoring nvitop-exporter
```

### View Pod Logs
```bash
kubectl logs -n monitoring -l app=nvitop-exporter
```

### Check GPU Access
```bash
kubectl exec -n monitoring -it <pod-name> -- nvidia-smi
```

### Verify Metrics
```bash
kubectl exec -n monitoring -it <pod-name> -- curl localhost:8080/metrics
```

## 🏷️ Labels and Selectors

The manifests use consistent labeling:
- `app: nvitop-exporter`
- `component: gpu-monitoring`
- `app.kubernetes.io/name: nvitop-exporter`
- `app.kubernetes.io/component: gpu-monitoring`
- `app.kubernetes.io/part-of: monitoring`

## 🔒 Security

- Runs as privileged container (required for GPU access)
- Uses dedicated ServiceAccount with minimal RBAC permissions
- Read-only access to host filesystems where possible

## 📊 Metrics Exposed

The exporter provides GPU metrics including:
- GPU utilization
- Memory usage
- Temperature
- Power consumption
- Process information
- And more...

Access metrics at: `http://<node-ip>:8080/metrics`
