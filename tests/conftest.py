# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

"""Shared fixtures for the recognizer test suite."""

import pytest

from nvitop.api import recognizer


@pytest.fixture(autouse=True)
def _isolated_recognizer():
    """Save and restore global recognizer state around every test."""
    engine_rules = recognizer.ENGINE_RULES[:]
    service_rules = recognizer.SERVICE_RULES[:]
    enabled = recognizer.enabled()
    platform = recognizer._PLATFORM
    recognizer._FACTS_CACHE.clear()
    recognizer._CONTAINER_NAME_CACHE.clear()
    recognizer._OLLAMA_NAME_CACHE.clear()
    yield
    recognizer.ENGINE_RULES[:] = engine_rules
    recognizer.SERVICE_RULES[:] = service_rules
    recognizer.set_enabled(enabled)
    recognizer._PLATFORM = platform
    recognizer._FACTS_CACHE.clear()
    recognizer._CONTAINER_NAME_CACHE.clear()
    recognizer._OLLAMA_NAME_CACHE.clear()
