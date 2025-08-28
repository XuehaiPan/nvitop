# Kubernetes Pod Monitoring with nvitop-exporter

This document describes the Kubernetes pod monitoring functionality added to nvitop-exporter, which extends GPU process metrics with pod information.

## 🚀 Features

The nvitop-exporter now automatically detects when running in a Kubernetes environment and enriches process metrics with pod information:

- **Pod Name**: The name of the Kubernetes pod running the GPU process
- **Pod Namespace**: The namespace of the pod
- **Container Name**: The specific container within the pod running the process

## 📊 Enhanced Metrics

All process-related metrics now include additional labels when Kubernetes pod information is available:

```
process_gpu_memory{hostname="node-1", index="0", devicename="Tesla V100", uuid="GPU-12345", 
                   pid="1234", username="root", pod_name="training-job-abc", 
                   pod_namespace="ml-workloads", container_name="pytorch"} 2048
```

### New Labels

| Label | Description | Example |
|-------|-------------|---------|
| `pod_name` | Kubernetes pod name | `training-job-abc` |
| `pod_namespace` | Pod namespace | `ml-workloads` |
| `container_name` | Container name within pod | `pytorch` |

## 🔧 Configuration

### Enable/Disable Kubernetes Integration

By default, Kubernetes integration is enabled when running in a Kubernetes environment. You can control it via CLI flags:

```bash
# Enable (default)
nvitop-exporter --enable-k8s

# Disable  
nvitop-exporter --disable-k8s
```

### Environment Detection

The exporter automatically detects Kubernetes environments by checking for:

1. Service account token: `/var/run/secrets/kubernetes.io/serviceaccount/token`
2. Environment variables: `KUBERNETES_SERVICE_HOST`, `KUBERNETES_SERVICE_PORT`

## 🐳 Kubernetes Deployment

The provided Kubernetes manifests in the `k8s/` directory are pre-configured for pod monitoring:

### Required Permissions

The DaemonSet includes ClusterRole permissions to read pods:

```yaml
rules:
- apiGroups: [""]
  resources: ["pods"] 
  verbs: ["get", "list", "watch"]
```

### Container Runtime Access

The DaemonSet mounts container runtime sockets to read container metadata:

```yaml
volumeMounts:
- name: docker-sock
  mountPath: /var/run/docker.sock
  readOnly: true
- name: containerd-sock
  mountPath: /run/containerd/containerd.sock
  readOnly: true
```

## 🔍 How It Works

### PID to Pod Mapping

The exporter uses a multi-step approach to map process IDs to Kubernetes pods:

1. **cgroup Analysis**: Reads `/proc/{pid}/cgroup` to extract pod UID and container ID
2. **Container Runtime**: Queries Docker/containerd for container labels containing pod metadata
3. **Kubernetes API**: Falls back to querying the Kubernetes API (future enhancement)

### Supported Container Runtimes

- **Docker**: Reads container configuration from `/var/lib/docker/containers/{id}/config.v2.json`
- **containerd**: Parses container task information (basic support)
- **CRI-O**: Supported via cgroup parsing

### cgroup Path Patterns

The exporter recognizes various cgroup path formats:

```
# cgroup v1
/kubepods/burstable/pod{uuid}/{container-id}
/kubepods/besteffort/pod{uuid}/{container-id}

# cgroup v2 (systemd)  
/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod{uuid}.slice/cri-containerd-{id}.scope
```

## 📈 Prometheus Queries

### GPU Usage by Pod

```promql
sum by (pod_name, pod_namespace) (process_gpu_memory{pod_name!=""})
```

### Top GPU-Using Pods

```promql
topk(10, sum by (pod_name, pod_namespace) (process_gpu_memory{pod_name!=""}))
```

### GPU Usage by Namespace

```promql
sum by (pod_namespace) (process_gpu_memory{pod_namespace!=""})
```

### Processes Without Pod Information

```promql
process_gpu_memory{pod_name=""}
```

## 🎯 Grafana Dashboard Examples

### Pod GPU Memory Table

Add a table panel with query:
```promql
sum by (pod_name, pod_namespace, container_name) (process_gpu_memory{pod_name!=""})
```

### Namespace GPU Utilization Chart  

Add a time series panel with query:
```promql
sum by (pod_namespace) (rate(process_gpu_sm_utilization{pod_namespace!=""}[5m]))
```

## 🛠️ Troubleshooting

### No Pod Information Appearing

1. **Check Kubernetes Detection**:
   ```bash
   kubectl logs -n monitoring <nvitop-exporter-pod> | grep -i kubernetes
   ```

2. **Verify Permissions**:
   ```bash
   kubectl auth can-i get pods --as=system:serviceaccount:monitoring:nvitop-exporter
   ```

3. **Check Container Runtime Mounts**:
   ```bash
   kubectl exec -n monitoring <pod> -- ls -la /var/run/docker.sock
   kubectl exec -n monitoring <pod> -- ls -la /run/containerd/containerd.sock
   ```

### Debugging cgroup Parsing

Check process cgroup information:
```bash
kubectl exec -n monitoring <pod> -- cat /host/proc/<pid>/cgroup
```

### Performance Considerations

- Pod information is cached per PID to reduce lookup overhead
- Enable only in Kubernetes environments to avoid unnecessary processing
- Consider increasing collection interval (`--interval`) for large clusters

## 🔒 Security Notes

- The DaemonSet runs with privileged access for GPU and container runtime access
- Container runtime socket access is read-only
- Service account has minimal required permissions for pod metadata access

## 🚦 Limitations

- Pod information is only available for processes visible via cgroup analysis
- Container runtime metadata parsing depends on runtime-specific formats  
- Some container runtimes may require additional volume mounts
- Cross-node pod information requires Kubernetes API access (future enhancement)