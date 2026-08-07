from __future__ import annotations

import json
import inspect
import time
import sys
import types
from contextlib import nullcontext

import numpy as np
import pytest

from eve.core.loop import (
    EVE_FIRST_EDITION_FIELDS,
    EVE_FIRST_EDITION_PROMPT,
    REPAIR_CONTEXT_MAX_TOKENS,
    REPAIR_MAX_NEW_TOKENS,
    TEXT_CONTEXT_MAX_TOKENS,
    TEXT_MAX_NEW_TOKENS,
    VISION_CONTEXT_MAX_TOKENS,
    VISION_MAX_NEW_TOKENS,
    ORGAN_NAMES,
    ORGAN_PROMPTS,
    CoreLoop,
    create_runtime_state,
)
import eve.core.loop as loop_module
from eve.input.buffer import InputBuffer
from eve.memory.memorizer import Memorizer


def first_edition_result(**updates):
    value = {
        "reply": "",
        "thinking_summary": "",
        "world_interpretation_update": {},
        "myself_cognition_update": {},
        "goodness_update": {},
        "goodness_records": [],
        "blackboard_updates": [],
        "active_tnn": [],
        "memory_actions": [],
        "training_proposal": None,
        "prompt_request": None,
        "action_candidates": [],
    }
    value.update(updates)
    return value


def test_fixed_prompt_is_first_edition_and_lists_only_organs():
    assert "EVE 第一版固定提示词" in EVE_FIRST_EDITION_PROMPT
    assert "protocol_version" not in EVE_FIRST_EDITION_PROMPT
    assert tuple(ORGAN_PROMPTS) == ORGAN_NAMES
    for name in ORGAN_NAMES:
        assert name in EVE_FIRST_EDITION_PROMPT
    assert "没有 Vision 或结构化视觉事实时" in EVE_FIRST_EDITION_PROMPT
    assert "没有 Output 成功反馈时" in EVE_FIRST_EDITION_PROMPT
    assert set(EVE_FIRST_EDITION_FIELDS) == {
        "action_candidates", "active_tnn", "blackboard_updates",
        "goodness_records", "goodness_update", "memory_actions",
        "myself_cognition_update", "prompt_request", "reply",
        "thinking_summary", "training_proposal",
        "world_interpretation_update",
    }
    assert (TEXT_CONTEXT_MAX_TOKENS, VISION_CONTEXT_MAX_TOKENS) == (4096, 4096)
    assert REPAIR_CONTEXT_MAX_TOKENS == 1536
    assert (TEXT_MAX_NEW_TOKENS, VISION_MAX_NEW_TOKENS, REPAIR_MAX_NEW_TOKENS) == (
        256, 160, 192
    )


def test_first_edition_shape_rejects_version_and_bounds_organ_prompt(tmp_path):
    core = CoreLoop(
        InputBuffer(), Memorizer(tmp_path / "memory"), log_dir=tmp_path,
        trainer=object(),
    )
    request = {"request_id": "task-1", "kind": "user", "message": "move"}
    clean = core._coerce_llm_result(
        first_edition_result(reply="need schema", prompt_request="mouse"), request
    )
    assert clean["prompt_request"] == "mouse"
    core._handle_prompt_request(request, "mouse")
    assert core._llm_requests.empty()
    assert core.state["blackboard"]["organ_prompt_rejected"]["value"]["reason"] == (
        "common_organ_schema_is_already_inline"
    )
    with pytest.raises(ValueError, match="unknown"):
        core._coerce_llm_result(
            {**first_edition_result(reply="x"), "protocol_version": 2}, request
        )
    for invalid in (
        {key: value for key, value in first_edition_result().items() if key != "reply"},
        {**first_edition_result(), "interpretation": {}},
        {**first_edition_result(), "unknown_field": None},
    ):
        with pytest.raises(ValueError, match="fields mismatch"):
            core._coerce_llm_result(invalid, request)


@pytest.mark.parametrize(
    ("action_type", "payload"),
    [
        ("mouse", {"action": "click", "x": 2, "y": 3}),
        ("keyboard", {"action": "write", "text": "你好", "method": "unicode"}),
        ("speak", {"text": "hello"}),
    ],
)
def test_body_candidates_are_normalized_with_core_owned_fields(
    tmp_path, action_type, payload
):
    core = CoreLoop(
        InputBuffer(), Memorizer(tmp_path / "memory"), log_dir=tmp_path,
        trainer=object(),
    )
    clean = core._coerce_llm_result(
        first_edition_result(
            reply="candidate",
            action_candidates=[
                {
                    "action_type": action_type,
                    "payload": payload,
                    "horizon_ms": 1000,
                    "reason_summary": "bounded",
                }
            ],
        ),
        {"request_id": "task", "kind": "user", "message": "act"},
    )
    candidate = clean["action_candidates"][0]
    assert candidate["candidate_id"].startswith("llm:")
    assert candidate["valid_for_ms"] == 1000
    assert candidate["action_type"] == action_type
    assert "action_id" not in candidate
    assert "generated_at_ns" not in candidate


def test_reply_does_not_create_speak_candidate(tmp_path):
    core = CoreLoop(InputBuffer(), Memorizer(tmp_path / "memory"), log_dir=tmp_path)
    clean = core._coerce_llm_result(
        first_edition_result(reply="GUI only"),
        {"request_id": "reply", "kind": "user", "message": "hello"},
    )
    assert clean["reply"] == "GUI only"
    assert clean["action_candidates"] == []


@pytest.mark.parametrize(
    ("action_type", "payload"),
    [
        ("mouse", {"action": "click", "x": 2, "y": 3}),
        ("keyboard", {"action": "press", "keys": ["A"]}),
        ("speak", {"text": "hello"}),
    ],
)
def test_body_candidates_pass_existing_queue_gate_and_mock_output(
    tmp_path, action_type, payload
):
    state = create_runtime_state(output_mode="mock", allow_mock_actions=True)
    state["cold_started"] = True
    memory = Memorizer(tmp_path / "memory")
    memory.start_writer()
    core = CoreLoop(InputBuffer(), memory, state=state, log_dir=tmp_path)
    candidate = core._coerce_action_candidates(
        [{"action_type": action_type, "payload": payload, "horizon_ms": 1000}]
    )[0]
    core._accept_llm_action_candidate(
        candidate, {"request_id": "action-task", "kind": "user", "message": "act"}
    )
    assert len(state["action_queue"]) == 1
    core.step()
    deadline = time.monotonic() + 2
    while state["latest_output"] is None and time.monotonic() < deadline:
        time.sleep(0.01)
    core._stop_output_worker(2)
    memory.stop_writer()
    assert state["latest_output"]["simulated"] is True
    assert state["latest_output"]["action_id"].startswith("action_")
    assert state["blackboard"]["latest_output_feedback"]["value"]["simulated"] is True


def test_body_candidate_is_rejected_when_permission_is_closed(tmp_path):
    state = create_runtime_state(output_mode="mock", allow_mock_actions=False)
    state["cold_started"] = True
    memory = Memorizer(tmp_path / "memory")
    memory.start_writer()
    core = CoreLoop(InputBuffer(), memory, state=state, log_dir=tmp_path)
    candidate = core._coerce_action_candidates(
        [{"action_type": "speak", "payload": {"text": "blocked"}, "horizon_ms": 1000}]
    )[0]
    core._accept_llm_action_candidate(
        candidate, {"request_id": "blocked", "kind": "user", "message": "act"}
    )
    results = core.step()
    core._stop_output_worker(2)
    memory.stop_writer()
    assert results[0]["blocked"] is True
    assert results[0]["reason"] == "permission_speak_not_allowed"
    assert "action_id" not in results[0]


def test_qwen_loader_creates_one_shared_model_and_processor(tmp_path, monkeypatch):
    calls = {"processor": 0, "model": 0}

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def mem_get_info():
            return (80, 100)

        @staticmethod
        def memory_allocated():
            return 10

        @staticmethod
        def memory_reserved():
            return 20

    class Parameter:
        device = "cuda:0"

    class Linear4bit:
        pass

    class Model:
        is_loaded_in_4bit = True
        hf_device_map = {"model": "cuda:0"}

        def parameters(self):
            return iter([Parameter()])

        def modules(self):
            return iter([self, Linear4bit()])

    class AutoProcessor:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            calls["processor"] += 1
            return object()

    class AutoModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            calls["model"] += 1
            return Model()

    torch = types.ModuleType("torch")
    torch.cuda = Cuda()
    torch.bfloat16 = object()
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = AutoProcessor
    transformers.AutoModelForImageTextToText = AutoModel
    transformers.AutoModelForVision2Seq = AutoModel
    def bits_config(**kwargs):
        calls["quantization"] = kwargs
        return object()

    transformers.BitsAndBytesConfig = bits_config
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    bitsandbytes = types.ModuleType("bitsandbytes")
    bitsandbytes.__path__ = []
    bitsandbytes_nn = types.ModuleType("bitsandbytes.nn")
    bitsandbytes_nn.Linear4bit = Linear4bit
    monkeypatch.setitem(sys.modules, "bitsandbytes", bitsandbytes)
    monkeypatch.setitem(sys.modules, "bitsandbytes.nn", bitsandbytes_nn)
    monkeypatch.setitem(sys.modules, "accelerate", types.ModuleType("accelerate"))

    qwen_path = tmp_path / "qwen"
    qwen_path.mkdir()
    core = CoreLoop(
        InputBuffer(), Memorizer(tmp_path / "memory"), log_dir=tmp_path,
        trainer=object(),
    )
    core.state["model_config"]["qwen_path"] = str(qwen_path)
    core._load_local_llm(str(qwen_path))
    model = core._qwen_model
    processor = core._qwen_processor
    core._load_vlm(str(qwen_path))

    assert calls["processor"] == 1
    assert calls["model"] == 1
    assert calls["quantization"]["load_in_4bit"] is True
    assert calls["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert calls["quantization"]["bnb_4bit_use_double_quant"] is True
    assert core._qwen_model is model
    assert core._qwen_processor is processor
    assert core.state["model_status"]["qwen"]["hf_device_map"] == {
        "model": "cuda:0"
    }
    assert core.state["model_status"]["qwen"]["linear4bit_count"] == 1


def test_qwen_4bit_verification_requires_flag_and_linear_module(tmp_path):
    class Linear4bit:
        pass

    class Model:
        def __init__(self, flag, modules):
            self.is_loaded_in_4bit = flag
            self._modules = modules

        def modules(self):
            return iter(self._modules)

    assert CoreLoop._verify_qwen_4bit(
        Model(True, [Linear4bit()]), Linear4bit
    ) == (True, 1)
    assert CoreLoop._verify_qwen_4bit(Model(True, []), Linear4bit) == (False, 0)
    assert CoreLoop._verify_qwen_4bit(
        Model(False, [Linear4bit()]), Linear4bit
    ) == (False, 1)


def test_text_only_uses_tokenizer_without_pixel_values(tmp_path, monkeypatch):
    seen = {}

    class Tensor(np.ndarray):
        def to(self, _device):
            return self

    def tensor(values):
        return np.asarray(values).view(Tensor)

    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, **_kwargs):
            seen["messages"] = messages
            return "text prompt"

        def __call__(self, prompt, **_kwargs):
            seen["prompt"] = prompt
            seen["tokenizer_kwargs"] = _kwargs
            return {"input_ids": tensor([[1, 2]]), "attention_mask": tensor([[1, 1]])}

        def decode(self, values, **_kwargs):
            seen["decoded"] = list(values)
            return "ok"

    class Processor:
        tokenizer = Tokenizer()

    class Parameter:
        device = "cpu"

    class Model:
        def parameters(self):
            return iter([Parameter()])

        def generate(self, **inputs):
            assert "pixel_values" not in inputs
            assert not any(name.startswith("image_") for name in inputs)
            seen["max_new_tokens"] = inputs["max_new_tokens"]
            return tensor([[1, 2, 3]])

    torch = types.ModuleType("torch")
    torch.inference_mode = nullcontext
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    transformers = types.ModuleType("transformers")
    transformers.StoppingCriteria = object
    transformers.StoppingCriteriaList = list
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    core = CoreLoop(
        InputBuffer(), Memorizer(tmp_path / "memory"), log_dir=tmp_path,
        trainer=object(),
    )
    core._qwen_processor = Processor()
    core._qwen_model = Model()
    assert core._generate_text_only(
        [{"role": "user", "content": "hello"}],
        max_new_tokens=64,
        suppress_reasoning=True,
    ) == "ok"
    assert seen["prompt"] == "text prompt"
    assert seen["tokenizer_kwargs"]["truncation"] is True
    assert seen["tokenizer_kwargs"]["max_length"] == 4096
    assert seen["max_new_tokens"] == 64


def test_context_budget_drops_low_priority_sections_without_cutting_json(tmp_path):
    class CharTokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return "\n".join(item["content"] for item in messages)

        def __call__(self, prompt, **_kwargs):
            return {"input_ids": list(range(len(prompt)))}

    core = CoreLoop(
        InputBuffer(), Memorizer(tmp_path / "memory"), log_dir=tmp_path,
        trainer=object(),
    )
    context = {
        "trigger_kind": "user",
        "current_message_or_task": "KEEP-CURRENT-USER-MESSAGE",
        "current_goal": "KEEP-CURRENT-GOAL",
        "world_view": {"changes_this_turn": {"new": "fact"}},
        "related_memory": [{"summary": "x" * 500} for _ in range(5)],
        "recent_conversation": [{"reply": "y" * 500} for _ in range(6)],
        "blackboard_view": [{"value": "z" * 500} for _ in range(10)],
        "memory_views": {"stm_ids": ["m"] * 100},
        "tnn_view": [{"purpose": "t" * 500} for _ in range(5)],
    }
    fitted = core._fit_context_to_token_budget(
        CharTokenizer(), "system", context, 500
    )
    rendered = json.dumps(fitted, ensure_ascii=False)
    assert "KEEP-CURRENT-USER-MESSAGE" in rendered
    assert "KEEP-CURRENT-GOAL" in rendered
    assert len(rendered) + len("system\n") <= 500
    json.loads(rendered)


def test_memory_context_uses_metadata_and_bounds_explicit_evidence(
    tmp_path, monkeypatch
):
    memory = Memorizer(tmp_path / "memory")
    ids = [
        memory.create({"secret": str(index) * 4000}, "observation")
        for index in range(7)
    ]
    core = CoreLoop(InputBuffer(), memory, log_dir=tmp_path, trainer=object())
    reads = []
    real_read = memory.read

    def tracked_read(memory_id):
        reads.append(memory_id)
        return real_read(memory_id)

    monkeypatch.setattr(memory, "read", tracked_read)
    automatic = core._llm_context(
        {"request_id": "auto", "kind": "user", "message": "current"}
    )
    assert reads == []
    assert len(automatic["related_memory"]) == 5
    assert all("payload" not in item for item in automatic["related_memory"])
    explicit = core._llm_context(
        {
            "request_id": "evidence", "kind": "user", "message": "current",
            "evidence_memory_ids": ids,
        }
    )
    assert reads == ids[:3]
    assert len(explicit["explicit_evidence"]) == 3
    assert all(
        len(json.dumps(item["payload"], ensure_ascii=False)) <= 1900
        for item in explicit["explicit_evidence"]
    )


def test_vision_generation_enforces_input_and_output_budgets(tmp_path, monkeypatch):
    seen = {}

    class Tensor(np.ndarray):
        def to(self, _device):
            return self

    def tensor(values):
        return np.asarray(values).view(Tensor)

    class Processor:
        def apply_chat_template(self, _messages, **kwargs):
            seen["template_kwargs"] = kwargs
            return {"input_ids": tensor([[1] * 4096])}

        def batch_decode(self, _values, **_kwargs):
            return ['{"summary":"ok","verified_detections":[],"corrections":[]}']

    class Parameter:
        device = "cpu"

    class Model:
        def parameters(self):
            return iter([Parameter()])

        def generate(self, **inputs):
            seen["generate_kwargs"] = inputs
            return tensor([[1] * 4097])

    torch = types.ModuleType("torch")
    torch.inference_mode = nullcontext
    transformers = types.ModuleType("transformers")
    transformers.StoppingCriteria = object
    transformers.StoppingCriteriaList = list
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    core = CoreLoop(
        InputBuffer(), Memorizer(tmp_path / "memory"), log_dir=tmp_path,
        trainer=object(),
    )
    core._qwen_processor = Processor()
    core._qwen_model = Model()
    result = core._generate_with_vision(
        {
            "image": np.zeros((4, 4, 4), dtype=np.uint8),
            "prompt": "look",
            "runtime_visual_result": None,
        }
    )
    assert '"summary":"ok"' in result
    assert seen["template_kwargs"]["enable_thinking"] is False
    assert seen["template_kwargs"]["tokenizer_kwargs"] == {
        "truncation": True, "max_length": 4096,
    }
    assert seen["generate_kwargs"]["max_new_tokens"] == 160
    assert seen["generate_kwargs"]["do_sample"] is False


def test_runtime_prompt_source_has_no_removed_protocol_or_continuation_prompts():
    source = inspect.getsource(loop_module)
    for removed in (
        "observation_completion", "training_materialization", "tool_requests",
        '"kind": "organ_prompt"', "perception_facts",
    ):
        assert removed not in source
