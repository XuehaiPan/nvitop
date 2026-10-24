# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

"""Tests for the generic engine rule: model identity of processes renamed to ``ENGINE::role``.

vLLM, SGLang, ollama's vLLM workers and friends all rename their processes with a
``setproctitle`` of the form ``ENGINE::role`` and hide the served model in the parent
chain. The rule is generic: the engine label comes from the title prefix, and the model
is extracted from the ancestors' deployment flags or model paths.
"""

import pytest

from nvitop.api.recognizer import ProcessFacts, engine_rule, recognize


def _facts(cmdline, ancestors=(), **kwargs):
    return ProcessFacts(cmdline=tuple(cmdline), ancestor_cmdlines=tuple(ancestors), **kwargs)


def _serve_ancestors(model_path):
    return (
        ('VLLM::ApiServer',),
        ('python', '-m', 'vllm.entrypoints.openai.api_server', '--model', model_path),
    )


# ------------------------------------------------------------------------------
# Model extraction from deployment command lines
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('ancestor', 'expected'),
    [
        (
            ('vllm', 'serve', '/models/Qwen2.5-72B-Instruct', '--port', '8000'),
            'Qwen2.5-72B-Instruct',
        ),
        (('vllm', 'serve', 'Qwen2.5-72B-Instruct'), 'Qwen2.5-72B-Instruct'),
        (('vllm', 'serve', '/models/Qwen2.5-72B-Instruct/'), 'Qwen2.5-72B-Instruct'),
        (
            ('/usr/local/bin/vllm', 'serve', '/models/Llama-3-70B', '--tensor-parallel-size', '2'),
            'Llama-3-70B',
        ),
        (
            (
                'python',
                '-m',
                'vllm.entrypoints.openai.api_server',
                '--model',
                '/models/Mixtral-8x7B',
            ),
            'Mixtral-8x7B',
        ),
        (
            ('python', '-m', 'vllm', 'serve', '/models/DeepSeek-R1-Distill-32B'),
            'DeepSeek-R1-Distill-32B',
        ),
    ],
)
def test_model_extracted_from_deployment_cmdlines(ancestor, expected):
    facts = _facts(('VLLM::EngineCore0',), ancestors=(ancestor,))
    assert recognize(facts) == f'vllm: {expected}'


def test_model_flag_equals_form():
    ancestor = ('vllm', 'serve', '--model=/models/GLM-4-9B')
    facts = _facts(('VLLM::EngineCore0',), ancestors=(ancestor,))
    assert recognize(facts) == 'vllm: GLM-4-9B'


def test_model_flag_takes_precedence_over_positional():
    ancestor = ('vllm', 'serve', '/models/Positional', '--model', '/models/Flagged')
    facts = _facts(('VLLM::EngineCore0',), ancestors=(ancestor,))
    assert recognize(facts) == 'vllm: Flagged'


def test_served_model_name_flag_wins_over_the_raw_model_path():
    ancestor = (
        'vllm',
        'serve',
        '/models/Qwen2.5-72B-Instruct',
        '--served-model-name',
        'qwen72b',
    )
    facts = _facts(('VLLM::EngineCore0',), ancestors=(ancestor,))
    assert recognize(facts) == 'vllm: qwen72b'


def test_underscore_style_flags_paddleocr_genai_server():
    ancestor = (
        '/usr/local/bin/python',
        '/usr/local/bin/paddleocr',
        'genai_server',
        '--model_name',
        'PaddleOCR-VL-1.6-0.9B',
        '--model_dir',
        '/home/paddleocr/.paddlex/official_models/PaddleOCR-VL-1.6',
        '--host',
        '127.0.0.1',
        '--backend',
        'vllm',
    )
    facts = _facts(('VLLM::EngineCore',), ancestors=(ancestor,))
    assert recognize(facts) == 'vllm: PaddleOCR-VL-1.6-0.9B'


# ------------------------------------------------------------------------------
# The rule is generic over engines: label = the ENGINE:: title prefix
# ------------------------------------------------------------------------------


def test_sglang_worker_with_model_flag():
    facts = _facts(
        ('sglang::scheduler_TP0',),
        ancestors=(('python', '-m', 'sglang.launch_server', '--model', '/models/D'),),
    )
    assert recognize(facts) == 'sglang: D'


def test_tgi_router_with_model_id_flag():
    facts = _facts(
        ('TGI::Router',),
        ancestors=(('text-generation-launcher', '--model-id', '/models/F'),),
    )
    assert recognize(facts) == 'tgi: F'


def test_generic_prefix_shows_the_title_engine():
    facts = _facts(('myworker::0',), ancestors=(('vllm', 'serve', '/models/E'),))
    assert recognize(facts) == 'myworker: E'


# ------------------------------------------------------------------------------
# Model paths without flags: ollama blobs, HF cache names, model files
# ------------------------------------------------------------------------------


OLLAMA_BLOB = 'a' * 64


def _ollama_manifests(monkeypatch, tmp_path, digest, name, tag):
    manifest_dir = tmp_path / 'manifests' / 'registry.ollama.ai' / 'library' / name
    manifest_dir.mkdir(parents=True)
    (manifest_dir / tag).write_text(f'{{"config": "{digest}"}}', encoding='utf-8')
    monkeypatch.setattr(
        'nvitop.api.recognizer._OLLAMA_MODEL_ROOTS',
        (str(tmp_path),),
    )


def test_ollama_blob_path_resolves_to_the_manifest_name(monkeypatch, tmp_path):
    _ollama_manifests(monkeypatch, tmp_path, f'sha256-{OLLAMA_BLOB}', 'qwen3', '32b')
    facts = _facts(
        ('VLLM::EngineCore',),
        ancestors=(
            (
                '/usr/local/bin/ollama',
                'runner',
                '--model',
                f'/usr/share/ollama/.ollama/models/blobs/sha256-{OLLAMA_BLOB}',
            ),
        ),
    )
    assert recognize(facts) == 'vllm: qwen3:32b'


def test_ollama_blob_without_a_matching_manifest_falls_back(monkeypatch, tmp_path):
    _ollama_manifests(monkeypatch, tmp_path, f'sha256-{"b" * 64}', 'other', 'model')
    facts = _facts(
        ('VLLM::EngineCore',),
        ancestors=(
            (
                '/usr/local/bin/ollama',
                'runner',
                '--model',
                f'/usr/share/ollama/.ollama/models/blobs/sha256-{OLLAMA_BLOB}',
            ),
        ),
    )
    assert recognize(facts) is None


def test_huggingface_cache_path_yields_the_repo_name():
    facts = _facts(
        ('VLLM::EngineCore',),
        ancestors=(
            (
                'python',
                'serve.py',
                '/home/u/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/abc123',
            ),
        ),
    )
    assert recognize(facts) == 'vllm: Qwen/Qwen2.5-7B'


def test_model_file_yields_the_file_name():
    facts = _facts(
        ('VLLM::EngineCore',),
        ancestors=(('llama-server', '-m', '/models/qwen/qwen3-32b-Q4.gguf'),),
    )
    assert recognize(facts) == 'vllm: qwen3-32b-Q4'


# ------------------------------------------------------------------------------
# Parent chain walking
# ------------------------------------------------------------------------------


def test_direct_parent_deployment():
    facts = _facts(('VLLM::EngineCore0',), ancestors=(('vllm', 'serve', '/models/A'),))
    assert recognize(facts) == 'vllm: A'


def test_deployment_found_beyond_renamed_intermediate_process():
    facts = _facts(
        ('VLLM::EngineCore0',),
        ancestors=(
            ('VLLM::ApiServer',),
            ('vllm', 'serve', '/models/B'),
        ),
    )
    assert recognize(facts) == 'vllm: B'


def test_worker_and_api_server_show_the_same_model():
    worker = _facts(('VLLM::EngineCore0',), ancestors=_serve_ancestors('/models/C'))
    api_server = _facts(('VLLM::ApiServer',), ancestors=(('vllm', 'serve', '/models/C'),))
    assert recognize(worker) == recognize(api_server) == 'vllm: C'


# ------------------------------------------------------------------------------
# Fall-back-to-original cases
# ------------------------------------------------------------------------------


def test_renamed_process_without_ancestors_is_not_recognized():
    assert recognize(_facts(('VLLM::EngineCore0',))) is None


def test_ancestors_without_a_model_are_not_recognized():
    facts = _facts(('VLLM::EngineCore0',), ancestors=((('bash', 'run.sh'),),))
    assert recognize(facts) is None


def test_engine_rule_does_not_match_untitled_processes():
    assert engine_rule(_facts(('python', '-m', 'vllm.entrypoints.openai.api_server'))) is None


def test_deployment_without_model_argument_is_not_recognized():
    facts = _facts(('VLLM::EngineCore0',), ancestors=(('vllm', 'serve', '--port', '8000'),))
    assert engine_rule(facts) is None


def test_title_without_the_double_colon_convention_is_not_recognized():
    facts = _facts(('myworker',), ancestors=(('vllm', 'serve', '/models/G'),))
    assert engine_rule(facts) is None
