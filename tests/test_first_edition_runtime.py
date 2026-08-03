from __future__ import annotations

import time
import sys
import types
from contextlib import nullcontext

import numpy as np
import pytest

from eve.core.loop import (
    EVE_FIRST_EDITION_PROMPT,
    ORGAN_NAMES,
    ORGAN_PROMPTS,
    CoreLoop,
    create_runtime_state,
)
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
    assert "没有图片输入时不得声称看见画面" in EVE_FIRST_EDITION_PROMPT
    assert "Output 成功反馈之前" in EVE_FIRST_EDITION_PROMPT


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
    continuation = core._llm_requests.get_nowait()
    assert continuation["root_request_id"] == "task-1"
    assert continuation["organ_prompt"] == ORGAN_PROMPTS["mouse"]
    core._handle_prompt_request(request, "mouse")
    assert core._llm_requests.empty()
    assert core.state["blackboard"]["organ_prompt_rejected"]["value"]["reason"] == (
        "duplicate_prompt_request"
    )
    with pytest.raises(ValueError, match="unknown"):
        core._coerce_llm_result(
            {**first_edition_result(reply="x"), "protocol_version": 2}, request
        )


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

    class Model:
        is_loaded_in_4bit = True
        hf_device_map = {"model": "cuda:0"}

        def parameters(self):
            return iter([Parameter()])

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
    transformers.BitsAndBytesConfig = lambda **_kwargs: object()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))
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

    assert calls == {"processor": 1, "model": 1}
    assert core._qwen_model is model
    assert core._qwen_processor is processor
    assert core.state["model_status"]["qwen"]["hf_device_map"] == {
        "model": "cuda:0"
    }


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
        max_new_tokens=4,
        suppress_reasoning=True,
    ) == "ok"
    assert seen["prompt"] == "text prompt"
