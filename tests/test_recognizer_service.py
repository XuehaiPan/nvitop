# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

"""Tests for the built-in service rule (which project owns a bare interpreter script)."""

from nvitop.api.recognizer import ProcessFacts, recognize


def _facts(cmdline, cwd=None):
    return ProcessFacts(cmdline=tuple(cmdline), cwd=cwd)


def test_python_script_with_directory():
    assert (
        recognize(_facts(('python', 'main.py', '--port', '18000'), '/home/airroot/octopus-api'))
        == 'main.py @ octopus-api'
    )


def test_absolute_script_path():
    assert (
        recognize(_facts(('python', '/srv/tools/export.py'), '/srv/tools')) == 'export.py @ tools'
    )


def test_python_versioned_interpreter():
    assert recognize(_facts(('python3.10', 'serve.py'), '/srv/api')) == 'serve.py @ api'


def test_python_dash_m_module():
    assert (
        recognize(_facts(('python', '-m', 'uvicorn', 'src.api.app:app'), '/srv/octopus'))
        == 'uvicorn @ octopus'
    )


def test_python_dash_m_dotted_module_shows_last_component():
    assert recognize(_facts(('python', '-m', 'pkg.sub.module'), '/srv/x')) == 'module @ x'


def test_script_without_cwd_is_not_recognized():
    assert recognize(_facts(('python', 'main.py'))) is None


def test_dash_m_without_module_is_not_recognized():
    assert recognize(_facts(('python', '-m'), '/srv')) is None


def test_non_python_interpreter_is_not_recognized():
    assert recognize(_facts(('bash', 'run.sh'), '/srv/project')) is None


def test_bare_interpreter_is_not_recognized():
    assert recognize(_facts(('python',), '/srv')) is None


def test_zombie_sentinel_is_not_recognized():
    assert recognize(_facts(('Zombie Process',), '/srv')) is None


def test_recognition_survives_root_directory_cwd():
    assert recognize(_facts(('python', 'main.py'), '/')) is None
