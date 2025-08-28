"""Kubernetes integration for nvitop-exporter."""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PodInfo:
    """Information about a Kubernetes pod."""
    name: str
    namespace: str
    container_name: str
    uid: str
    
    @property
    def full_name(self) -> str:
        """Get the full pod name including namespace."""
        return f"{self.namespace}/{self.name}"


class KubernetesHelper:
    """Helper class for Kubernetes pod information retrieval."""
    
    def __init__(self, enable_k8s: bool = True) -> None:
        """Initialize Kubernetes helper.
        
        Args:
            enable_k8s: Whether to enable Kubernetes pod detection.
        """
        self.enable_k8s = enable_k8s
        self._pid_to_pod_cache: Dict[int, Optional[PodInfo]] = {}
        self._in_kubernetes = self._detect_kubernetes_environment()
        
    def _detect_kubernetes_environment(self) -> bool:
        """Detect if running in a Kubernetes environment."""
        if not self.enable_k8s:
            return False
            
        # Check for Kubernetes service account token
        service_account_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
        if service_account_path.exists():
            return True
            
        # Check for Kubernetes environment variables
        k8s_env_vars = [
            "KUBERNETES_SERVICE_HOST",
            "KUBERNETES_SERVICE_PORT", 
            "KUBERNETES_PORT"
        ]
        
        return any(os.environ.get(var) for var in k8s_env_vars)
    
    def get_pod_info_from_pid(self, pid: int) -> Optional[PodInfo]:
        """Get pod information for a given process ID.
        
        Args:
            pid: Process ID to look up.
            
        Returns:
            PodInfo if the process belongs to a Kubernetes pod, None otherwise.
        """
        if not self._in_kubernetes:
            return None
            
        # Check cache first
        if pid in self._pid_to_pod_cache:
            return self._pid_to_pod_cache[pid]
            
        pod_info = self._extract_pod_info_from_cgroup(pid)
        self._pid_to_pod_cache[pid] = pod_info
        return pod_info
    
    def _extract_pod_info_from_cgroup(self, pid: int) -> Optional[PodInfo]:
        """Extract pod information from process cgroup.
        
        This method reads the /proc/{pid}/cgroup file to extract Kubernetes
        pod information from the cgroup hierarchy.
        """
        try:
            cgroup_path = f"/host/proc/{pid}/cgroup"
            if not os.path.exists(cgroup_path):
                # Fallback for non-containerized environments
                cgroup_path = f"/proc/{pid}/cgroup"
                
            with open(cgroup_path, 'r') as f:
                cgroup_content = f.read()
                
            return self._parse_cgroup_for_pod_info(cgroup_content)
            
        except (IOError, OSError, PermissionError):
            return None
    
    def _parse_cgroup_for_pod_info(self, cgroup_content: str) -> Optional[PodInfo]:
        """Parse cgroup content to extract pod information.
        
        Kubernetes cgroup paths typically follow patterns like:
        - /kubepods/burstable/pod<pod-uid>/<container-id>
        - /kubepods/besteffort/pod<pod-uid>/<container-id>  
        - /kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod<pod-uid>.slice/...
        """
        # Pattern for cgroup v1 (legacy)
        v1_patterns = [
            r'/kubepods(?:\.slice)?/(?:burstable|besteffort|guaranteed)/pod([a-f0-9\-]{36})/([a-f0-9]{64})',
            r'/kubepods(?:\.slice)?/pod([a-f0-9\-]{36})/([a-f0-9]{64})',
        ]
        
        # Pattern for cgroup v2 (systemd)  
        v2_patterns = [
            r'/kubepods\.slice/kubepods-(?:burstable|besteffort|guaranteed)\.slice/kubepods-(?:burstable|besteffort|guaranteed)-pod([a-f0-9_\-]{36})\.slice/cri-containerd-([a-f0-9]{64})\.scope',
            r'/kubepods\.slice/kubepods-pod([a-f0-9_\-]{36})\.slice/cri-containerd-([a-f0-9]{64})\.scope',
        ]
        
        all_patterns = v1_patterns + v2_patterns
        
        for line in cgroup_content.splitlines():
            for pattern in all_patterns:
                match = re.search(pattern, line)
                if match:
                    pod_uid_raw = match.group(1)
                    container_id = match.group(2)
                    
                    # Convert pod UID format (replace underscores with hyphens for systemd)
                    pod_uid = pod_uid_raw.replace('_', '-')
                    
                    # Try to get additional pod info via procfs or K8s API
                    pod_info = self._resolve_pod_details(pod_uid, container_id)
                    if pod_info:
                        return pod_info
                        
        return None
    
    def _resolve_pod_details(self, pod_uid: str, container_id: str) -> Optional[PodInfo]:
        """Resolve full pod details from UID and container ID.
        
        This method attempts to resolve pod name, namespace, and container name
        using various methods including reading from the container runtime.
        """
        # Method 1: Try to read from container labels (containerd/docker)
        pod_info = self._get_pod_info_from_container_labels(container_id)
        if pod_info:
            return pod_info
            
        # Method 2: Create a basic pod info with UID only
        # In production, you might want to query the Kubernetes API here
        return PodInfo(
            name=f"pod-{pod_uid[:8]}",  # Short form of UID
            namespace="unknown",
            container_name="unknown", 
            uid=pod_uid
        )
    
    def _get_pod_info_from_container_labels(self, container_id: str) -> Optional[PodInfo]:
        """Get pod information from container runtime labels.
        
        This method reads container information from the runtime to extract
        Kubernetes labels that contain pod metadata.
        """
        try:
            # Try containerd runtime first
            containerd_path = f"/run/containerd/io.containerd.runtime.v2.task/k8s.io/{container_id[:12]}"
            if os.path.exists(containerd_path):
                return self._parse_containerd_labels(containerd_path, container_id)
                
            # Try docker runtime
            docker_inspect_path = f"/var/lib/docker/containers/{container_id}/config.v2.json"
            if os.path.exists(docker_inspect_path):
                return self._parse_docker_labels(docker_inspect_path)
                
        except (IOError, OSError, PermissionError):
            pass
            
        return None
    
    def _parse_containerd_labels(self, containerd_path: str, container_id: str) -> Optional[PodInfo]:
        """Parse containerd container labels."""
        # This is a simplified implementation
        # In practice, you might need to parse containerd's task info
        return None
    
    def _parse_docker_labels(self, config_path: str) -> Optional[PodInfo]:
        """Parse Docker container configuration for Kubernetes labels."""
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            labels = config.get('Config', {}).get('Labels', {})
            
            pod_name = labels.get('io.kubernetes.pod.name')
            pod_namespace = labels.get('io.kubernetes.pod.namespace') 
            container_name = labels.get('io.kubernetes.container.name')
            pod_uid = labels.get('io.kubernetes.pod.uid')
            
            if all([pod_name, pod_namespace, container_name, pod_uid]):
                return PodInfo(
                    name=pod_name,
                    namespace=pod_namespace,
                    container_name=container_name,
                    uid=pod_uid
                )
        except (IOError, OSError, json.JSONDecodeError, KeyError):
            pass
            
        return None
    
    def clear_cache(self) -> None:
        """Clear the PID to pod mapping cache."""
        self._pid_to_pod_cache.clear()
    
    @property 
    def is_kubernetes_enabled(self) -> bool:
        """Check if Kubernetes integration is enabled and available."""
        return self._in_kubernetes and self.enable_k8s