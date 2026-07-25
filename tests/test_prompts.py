"""Tests for EVE Prompt System."""

import json

from eve.core.prompts import (
    world_update_prompt,
    myself_update_prompt,
    active_tnn_selection_prompt,
    memory_retrieval_prompt,
    training_order_prompt,
    teacher_prompt_generation_prompt,
    sleep_consolidation_prompt,
    validate_llm_output,
    _parse_json_from_text,
)


# ── Prompt String Contains Expected Keys ──────────────────────

def test_world_update_prompt_has_keys() -> None:
    text = world_update_prompt("world", "me", "bb", "hormones", "events")
    # Extract the JSON template from the prompt
    parsed = _parse_json_from_text(text)
    # The prompt contains a JSON template; let's verify keys are in the prompt text
    assert "scene" in text
    assert "sub_scene" in text
    assert "active_window" in text
    assert "visible_objects" in text
    assert "detected_text" in text
    assert "changes" in text
    assert "uncertainty" in text
    assert "attention_focus" in text


def test_active_tnn_prompt_has_keys() -> None:
    text = active_tnn_selection_prompt("world", "me", "available", "loaded", "hormones")
    assert "active_tnn" in text
    assert "reasoning" in text


def test_training_order_prompt_structure() -> None:
    text = training_order_prompt("failures", "loads", "hormones", "available")
    assert "should_train" in text
    assert "suggestion" in text
    assert "reason" in text
    assert "training_data_hint" in text
    assert "teacher" in text
    assert "priority" in text


def test_sleep_consolidation_prompt_structure() -> None:
    text = sleep_consolidation_prompt("today", "fails", "succ", "hist", "stats", "queue")
    assert "key_learnings" in text
    assert "failures_analysis" in text
    assert "success_patterns" in text
    assert "memory_consolidation" in text
    assert "skill_candidates" in text
    assert "training_recommendations" in text
    assert "self_adjustment" in text


def test_myself_update_prompt_structure() -> None:
    text = myself_update_prompt("world", "me", "hormones", "tasks", "tnn", "results")
    assert "what_im_thinking" in text
    assert "current_task" in text
    assert "task_progress" in text
    assert "confidence" in text
    assert "concerns" in text
    assert "suggestions" in text


# ── validate_llm_output ───────────────────────────────────────

def test_validate_llm_output_valid() -> None:
    schema = {"scene": "str", "visible_objects": "list", "should_train": "bool"}
    output = {"scene": "desktop", "visible_objects": ["icon1", "icon2"], "should_train": True}
    valid, err = validate_llm_output(output, schema)
    assert valid is True
    assert err == ""


def test_validate_llm_output_missing_key() -> None:
    schema = {"scene": "str", "visible_objects": "list"}
    output = {"scene": "desktop"}
    valid, err = validate_llm_output(output, schema)
    assert valid is False
    assert "missing key" in err
    assert "visible_objects" in err


def test_validate_llm_output_wrong_type() -> None:
    schema = {"scene": "str", "visible_objects": "list"}
    output = {"scene": "desktop", "visible_objects": "not_a_list"}
    valid, err = validate_llm_output(output, schema)
    assert valid is False
    assert "list" in err


def test_validate_llm_output_number_type() -> None:
    schema = {"confidence": "number"}
    assert validate_llm_output({"confidence": 0.5}, schema)[0] is True
    assert validate_llm_output({"confidence": 42}, schema)[0] is True
    assert validate_llm_output({"confidence": "high"}, schema)[0] is False
    assert validate_llm_output({}, schema)[0] is False


# ── _parse_json_from_text ────────────────────────────────────

def test_parse_json_from_code_block() -> None:
    text = '```json\n{"key": "value", "num": 42}\n```'
    result = _parse_json_from_text(text)
    assert result is not None
    assert result["key"] == "value"
    assert result["num"] == 42


def test_parse_json_bare() -> None:
    text = 'some text {"key": "value"} more text'
    result = _parse_json_from_text(text)
    assert result is not None
    assert result["key"] == "value"


def test_parse_json_invalid() -> None:
    text = "this is not json at all"
    result = _parse_json_from_text(text)
    assert result is None

    text2 = '{"key": "value"'
    result2 = _parse_json_from_text(text2)
    assert result2 is None
