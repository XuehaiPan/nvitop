#!/usr/bin/env python3
"""Test script for Kubernetes pod monitoring functionality."""

import os
import tempfile
from nvitop_exporter.k8s import KubernetesHelper, PodInfo


def create_mock_cgroup_file(content: str) -> str:
    """Create a temporary cgroup file for testing."""
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return path


def test_cgroup_v1_parsing():
    """Test cgroup v1 format parsing."""
    print("Testing cgroup v1 parsing...")
    
    k8s_helper = KubernetesHelper(enable_k8s=True)
    
    # Mock cgroup v1 content
    cgroup_content = """12:pids:/kubepods/burstable/pod12345678-1234-1234-1234-123456789abc/1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
11:net_cls,net_prio:/kubepods/burstable/pod12345678-1234-1234-1234-123456789abc/1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
10:freezer:/kubepods/burstable/pod12345678-1234-1234-1234-123456789abc/1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"""
    
    pod_info = k8s_helper._parse_cgroup_for_pod_info(cgroup_content)
    
    if pod_info:
        print(f"✅ Found pod info: {pod_info.name} in {pod_info.namespace}")
        print(f"   Container: {pod_info.container_name}, UID: {pod_info.uid}")
    else:
        print("❌ No pod info found")


def test_cgroup_v2_parsing():
    """Test cgroup v2 (systemd) format parsing."""
    print("\nTesting cgroup v2 parsing...")
    
    k8s_helper = KubernetesHelper(enable_k8s=True)
    
    # Mock cgroup v2 content  
    cgroup_content = """0::/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod12345678_1234_1234_1234_123456789abc.slice/cri-containerd-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef.scope"""
    
    pod_info = k8s_helper._parse_cgroup_for_pod_info(cgroup_content)
    
    if pod_info:
        print(f"✅ Found pod info: {pod_info.name} in {pod_info.namespace}")
        print(f"   Container: {pod_info.container_name}, UID: {pod_info.uid}")
    else:
        print("❌ No pod info found")


def test_kubernetes_detection():
    """Test Kubernetes environment detection."""
    print("\nTesting Kubernetes environment detection...")
    
    # Test with K8s disabled
    k8s_helper_disabled = KubernetesHelper(enable_k8s=False)
    print(f"K8s disabled: {k8s_helper_disabled.is_kubernetes_enabled}")
    
    # Test with K8s enabled but no environment  
    k8s_helper_enabled = KubernetesHelper(enable_k8s=True)
    print(f"K8s enabled, detected environment: {k8s_helper_enabled._in_kubernetes}")
    
    # Test with mock environment variable
    os.environ['KUBERNETES_SERVICE_HOST'] = 'kubernetes.default.svc'
    k8s_helper_env = KubernetesHelper(enable_k8s=True)
    print(f"With K8s env var: {k8s_helper_env._in_kubernetes}")
    
    # Clean up
    del os.environ['KUBERNETES_SERVICE_HOST']


def test_pod_info_creation():
    """Test PodInfo dataclass."""
    print("\nTesting PodInfo creation...")
    
    pod = PodInfo(
        name="test-pod-123",
        namespace="default", 
        container_name="main",
        uid="12345678-1234-1234-1234-123456789abc"
    )
    
    print(f"✅ Pod: {pod.full_name}")
    print(f"   Container: {pod.container_name}")
    print(f"   UID: {pod.uid}")


def test_cache_functionality():
    """Test PID caching functionality."""
    print("\nTesting cache functionality...")
    
    k8s_helper = KubernetesHelper(enable_k8s=True)
    
    # Mock a PID lookup (this would normally fail but we can test caching logic)
    pid = 1234
    
    # First lookup - should not be cached
    result1 = k8s_helper.get_pod_info_from_pid(pid)
    print(f"First lookup result: {result1}")
    
    # Second lookup - should be from cache  
    result2 = k8s_helper.get_pod_info_from_pid(pid)
    print(f"Second lookup result (cached): {result2}")
    
    # Clear cache
    k8s_helper.clear_cache()
    print("✅ Cache cleared")
    
    # Third lookup - should not be cached again
    result3 = k8s_helper.get_pod_info_from_pid(pid)
    print(f"Third lookup result (after cache clear): {result3}")


if __name__ == "__main__":
    print("🧪 nvitop-exporter Kubernetes Integration Tests")
    print("=" * 50)
    
    test_kubernetes_detection()
    test_pod_info_creation()
    test_cgroup_v1_parsing()
    test_cgroup_v2_parsing()
    test_cache_functionality()
    
    print("\n✅ All tests completed!")