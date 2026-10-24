# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

"""Tests for container attribution: cgroup detection, name resolution, composition."""

from nvitop.api import recognizer
from nvitop.api.recognizer import Container, _container_of, _detect_container


DOCKER_ID = 'a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890'

DOCKER_CGROUP_V1 = f'12:pids:/docker/{DOCKER_ID}\n13:cpuset:/docker/{DOCKER_ID}\n'
DOCKER_CGROUP_V2 = f'0::/system.slice/docker-{DOCKER_ID}.scope\n'
DOCKER_CGROUPFS_V2 = f'0::/docker/{DOCKER_ID}\n'
LXD_CGROUP = '10:cpuset:/lxc.payload.tts-box\n0::/lxc.payload.tts-box\n'
K8S_CGROUP = (
    '0::/kubepods.slice/kubepods-burstable.slice/'
    f'kubepods-burstable-pod123.slice/docker-{DOCKER_ID}.scope\n'
)
UNRELATED_CGROUP = '0::/user.slice/user-1000.slice/session-2.scope\n'


# ------------------------------------------------------------------------------
# cgroup detection
# ------------------------------------------------------------------------------


def test_detect_docker_cgroup_v1():
    assert _detect_container(DOCKER_CGROUP_V1) == Container(runtime='docker', identifier=DOCKER_ID)


def test_detect_docker_cgroup_v2_systemd_scope():
    assert _detect_container(DOCKER_CGROUP_V2) == Container(runtime='docker', identifier=DOCKER_ID)


def test_detect_docker_cgroup_v2_cgroupfs():
    assert _detect_container(DOCKER_CGROUPFS_V2) == Container(
        runtime='docker',
        identifier=DOCKER_ID,
    )


def test_detect_lxd_payload():
    assert _detect_container(LXD_CGROUP) == Container(runtime='lxd', identifier='tts-box')


def test_kubernetes_pods_are_ignored():
    assert _detect_container(K8S_CGROUP) is None


def test_unrelated_cgroups_are_ignored():
    assert _detect_container(UNRELATED_CGROUP) is None
    assert _detect_container(None) is None
    assert _detect_container('') is None


# ------------------------------------------------------------------------------
# Display names
# ------------------------------------------------------------------------------


def test_display_name_prefers_the_resolved_name():
    container = Container(runtime='docker', identifier=DOCKER_ID, name='vllm-svc')
    assert container.display_name == 'vllm-svc'


def test_display_name_falls_back_to_the_short_identifier():
    container = Container(runtime='docker', identifier=DOCKER_ID)
    assert container.display_name == DOCKER_ID[:12]


def test_display_name_for_lxd_uses_the_payload_name():
    container = Container(runtime='lxd', identifier='tts-box')
    assert container.display_name == 'tts-box'


# ------------------------------------------------------------------------------
# Detection plus name resolution (the glue that reads /proc and the Docker CLI)
# ------------------------------------------------------------------------------


def _with_cgroup(monkeypatch, cgroup):
    monkeypatch.setattr(recognizer, '_read_cgroup', lambda pid: cgroup)


def test_lxd_cgroup_carries_the_container_name(monkeypatch):
    _with_cgroup(monkeypatch, LXD_CGROUP)
    container = _container_of(4321)
    assert container is not None
    assert container.display_name == 'tts-box'


def test_docker_name_is_looked_up(monkeypatch):
    _with_cgroup(monkeypatch, DOCKER_CGROUP_V1)
    monkeypatch.setattr(recognizer, '_docker_container_name', lambda cid: 'vllm-svc')
    container = _container_of(4321)
    assert container is not None
    assert container.name == 'vllm-svc'
    assert container.display_name == 'vllm-svc'


def test_docker_name_lookup_is_cached_per_container_id(monkeypatch):
    _with_cgroup(monkeypatch, DOCKER_CGROUP_V1)
    calls = []

    def counting_lookup(cid):
        calls.append(cid)
        return 'vllm-svc'

    monkeypatch.setattr(recognizer, '_docker_container_name', counting_lookup)
    assert _container_of(4321) == Container(runtime='docker', identifier=DOCKER_ID, name='vllm-svc')
    assert _container_of(4322) == Container(runtime='docker', identifier=DOCKER_ID, name='vllm-svc')
    assert calls == [DOCKER_ID]


def test_docker_name_lookup_failure_falls_back_to_the_short_id(monkeypatch):
    _with_cgroup(monkeypatch, DOCKER_CGROUP_V2)
    monkeypatch.setattr(recognizer, '_docker_container_name', lambda cid: None)
    container = _container_of(4321)
    assert container is not None
    assert container.name is None
    assert container.display_name == DOCKER_ID[:12]


def test_process_outside_a_container(monkeypatch):
    _with_cgroup(monkeypatch, UNRELATED_CGROUP)
    assert _container_of(4321) is None


def test_unreadable_cgroup_file_yields_no_container(monkeypatch):
    monkeypatch.setattr(recognizer, '_read_cgroup', lambda pid: None)
    assert _container_of(4321) is None
    assert recognizer._read_cgroup(999999) is None  # the real reader on a gone PID


# ------------------------------------------------------------------------------
# End-to-end through the glue entry with real cgroup fixtures
# ------------------------------------------------------------------------------


class FakeProcess:
    pid = 4321

    def cmdline(self):
        return ['python', 'main.py']

    def parent(self):
        return None

    def cwd(self):
        return '/workspace'


def _linux(monkeypatch, cgroup, docker_name=None):
    recognizer._PLATFORM = 'linux'
    monkeypatch.setattr(recognizer, '_read_cgroup', lambda pid: cgroup)
    monkeypatch.setattr(recognizer, '_docker_container_name', lambda cid: docker_name)


def test_docker_process_with_resolved_name(monkeypatch):
    _linux(monkeypatch, DOCKER_CGROUP_V1, docker_name='vllm-svc')
    assert recognizer.recognize_command(FakeProcess()) == 'main.py @ vllm-svc:/workspace'


def test_docker_process_without_resolved_name_uses_short_id(monkeypatch):
    _linux(monkeypatch, DOCKER_CGROUP_V2, docker_name=None)
    expected = f'main.py @ {DOCKER_ID[:12]}:/workspace'
    assert recognizer.recognize_command(FakeProcess()) == expected


def test_lxd_process_uses_payload_name(monkeypatch):
    _linux(monkeypatch, LXD_CGROUP)
    assert recognizer.recognize_command(FakeProcess()) == 'main.py @ tts-box:/workspace'


def test_non_container_process_uses_bare_directory(monkeypatch):
    _linux(monkeypatch, UNRELATED_CGROUP)
    assert recognizer.recognize_command(FakeProcess()) == 'main.py @ workspace'
