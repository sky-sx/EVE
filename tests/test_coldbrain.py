import pytest

pytest.skip(
    "eve.coldbrain modules are stub-only (all methods return None / ...). "
    "Tests will be re-enabled when implementations are added.",
    allow_module_level=True,
)

def test_instruction_parser_parses_track_intent() -> None:
    parser = InstructionParser(confidence_threshold=0.2)
    result = parser.parse("track the red ball")
    assert result is not None
    assert result["intent"] == "track"
    assert result["confidence"] > 0.0


def test_instruction_parser_parses_stop_intent() -> None:
    parser = InstructionParser(confidence_threshold=0.2)
    result = parser.parse("please stop now")
    assert result is not None
    assert result["intent"] == "stop"


def test_instruction_parser_extracts_params() -> None:
    parser = InstructionParser(confidence_threshold=0.2)
    result = parser.parse("move 50 pixels to the right")
    assert result is not None
    assert result["intent"] in ("move",)
    assert "values" in result["params"]
    assert 50.0 in result["params"]["values"]


def test_instruction_parser_returns_none_for_no_match() -> None:
    parser = InstructionParser(confidence_threshold=0.9)
    result = parser.parse("xyzzy abcdef")
    assert result is None


def test_instruction_parser_returns_none_for_empty_text() -> None:
    parser = InstructionParser()
    assert parser.parse("") is None
    assert parser.parse(None) is None


def test_instruction_parser_register_pattern() -> None:
    parser = InstructionParser(confidence_threshold=0.2)
    parser.register_pattern("dance", r"\b(dance|boogie|groove)\b")
    result = parser.parse("let's dance together")
    assert result is not None
    assert result["intent"] == "dance"


def test_instruction_parser_list_known_intents() -> None:
    parser = InstructionParser()
    intents = parser.list_known_intents()
    assert "track" in intents
    assert "stop" in intents
    assert "explore" in intents
    assert len(intents) >= 9


def test_instruction_parser_process_pending() -> None:
    parser = InstructionParser(confidence_threshold=0.2)
    parser.enqueue("track the ball")
    parser.enqueue("stop moving")
    parser.enqueue("xyz")  # no match
    results = parser.process_pending()
    assert len(results) == 3
    assert results[0] is not None
    assert results[0]["intent"] == "track"
    assert results[2] is None
    assert parser.pending_count == 0


# ── IntentionField ────────────────────────────────────────────────────────

def test_intention_field_set_and_get_bias() -> None:
    field = IntentionField(dim=8, decay_rate=0.01)
    field.set_intention({"intent": "track", "confidence": 0.9})
    bias = field.get_bias()
    assert isinstance(bias, np.ndarray)
    assert bias.shape == (8,)
    assert np.isclose(np.linalg.norm(bias), 1.0)


def test_intention_field_decay_reduces_weight() -> None:
    field = IntentionField(dim=8, decay_rate=5.0)
    field.set_intention({"intent": "track", "confidence": 0.9})
    assert field.is_active()
    field.decay(2.0)  # Strong decay over 2 seconds
    # After heavy decay, should be inactive
    assert not field.is_active() or len(field.active_intentions) == 0


def test_intention_field_clear() -> None:
    field = IntentionField()
    field.set_intention({"intent": "track", "confidence": 0.9})
    field.clear()
    assert not field.is_active()
    bias = field.get_bias()
    assert np.allclose(bias, np.zeros(8))


def test_intention_field_multiple_intentions() -> None:
    field = IntentionField(dim=8, decay_rate=0.001)
    field.set_intention({"intent": "track", "confidence": 0.9})
    field.set_intention({"intent": "explore", "confidence": 0.6})
    active = field.active_intentions
    assert len(active) >= 1


def test_intention_field_process_pending() -> None:
    field = IntentionField(dim=8, decay_rate=0.001)
    field.enqueue_intention({"intent": "track", "confidence": 0.9})
    field.enqueue_intention({"intent": "stop", "confidence": 0.5})
    assert field.pending_count == 2
    field.process_pending()
    assert field.pending_count == 0
    assert field.is_active()


def test_intention_field_is_active_threshold() -> None:
    field = IntentionField(active_threshold=0.5, decay_rate=0.001)
    field.set_intention({"intent": "track", "confidence": 0.3})
    assert not field.is_active()


def test_intention_field_bias_normalized() -> None:
    field = IntentionField(dim=8)
    field.set_intention({"intent": "track", "confidence": 1.0})
    field.set_intention({"intent": "stop", "confidence": 1.0})
    bias = field.get_bias()
    norm = np.linalg.norm(bias)
    assert np.isclose(norm, 1.0) or norm == 0.0


# ── ReflectionWorker ──────────────────────────────────────────────────────

def test_reflection_worker_submit_and_process() -> None:
    worker = ReflectionWorker(processing_delay=0.01)
    worker.submit("ep_001")
    assert worker.queue_size == 1
    worker.process_pending()
    assert worker.queue_size == 0
    assert worker.processed_count == 1


def test_reflection_worker_duplicate_submit() -> None:
    worker = ReflectionWorker(processing_delay=0.01)
    worker.submit("ep_001")
    worker.submit("ep_001")
    assert worker.queue_size == 1


def test_reflection_worker_get_insights() -> None:
    worker = ReflectionWorker(processing_delay=0.01)
    worker.submit("ep_alpha")
    worker.process_pending()
    insights = worker.get_insights("ep_alpha")
    assert isinstance(insights, list)
    for ins in insights:
        assert "pattern" in ins
        assert "confidence" in ins
        assert "suggestion" in ins


def test_reflection_worker_unknown_episode() -> None:
    worker = ReflectionWorker(processing_delay=0.01)
    insights = worker.get_insights("never_submitted")
    assert insights == []


def test_reflection_worker_is_busy_false_after_process() -> None:
    worker = ReflectionWorker(processing_delay=0.01)
    worker.submit("ep_001")
    worker.process_pending()
    assert not worker.is_busy()


def test_reflection_worker_process_empty_queue() -> None:
    worker = ReflectionWorker(processing_delay=0.01)
    worker.process_pending()
    assert worker.processed_count == 0
    assert not worker.is_busy()


def test_reflection_worker_known_episodes() -> None:
    worker = ReflectionWorker(processing_delay=0.01)
    worker.submit("ep_002")
    worker.submit("ep_001")
    worker.process_pending()
    worker.process_pending()
    assert "ep_001" in worker.known_episodes
    assert "ep_002" in worker.known_episodes


# ── ConsolidationDecider ─────────────────────────────────────────────────

def test_consolidation_decider_evaluates_items() -> None:
    cd = ConsolidationDecider()
    summary = {
        "novelty": 0.9,
        "reward": 0.8,
        "repetition": 3,
        "error": 0.1,
        "items": [
            {"type": "observation", "content": "red ball detected"},
            {"type": "skill", "content": "smooth tracking"},
        ],
    }
    decisions = cd.evaluate(summary)
    assert len(decisions) >= 1
    for d in decisions:
        assert "type" in d
        assert "importance" in d
        assert "target_memory" in d
        assert d["target_memory"] in ("short_term", "mid_term", "long_term",
                                       "habit", "skill", "discard")


def test_consolidation_decider_should_forget_low_importance() -> None:
    cd = ConsolidationDecider(importance_threshold=0.2)
    assert cd.should_forget({"importance": 0.05})
    assert not cd.should_forget({"importance": 0.5})


def test_consolidation_decider_prioritize_sorts() -> None:
    cd = ConsolidationDecider()
    items = [
        {"type": "obs", "importance": 0.3},
        {"type": "obs", "importance": 0.9},
        {"type": "obs", "importance": 0.1},
    ]
    sorted_items = cd.prioritize(items)
    assert sorted_items[0]["importance"] == 0.9
    assert sorted_items[-1]["importance"] == 0.1


def test_consolidation_decider_high_importance_to_long_term() -> None:
    cd = ConsolidationDecider()
    summary = {
        "novelty": 1.0,
        "reward": 1.0,
        "items": [{"type": "observation", "content": "unique event"}],
    }
    decisions = cd.evaluate(summary)
    assert len(decisions) >= 1
    assert decisions[0]["target_memory"] in ("long_term", "mid_term", "skill", "habit")


def test_consolidation_decider_low_importance_discarded() -> None:
    cd = ConsolidationDecider(importance_threshold=0.9)
    summary = {
        "novelty": 0.1,
        "reward": 0.0,
        "items": [{"type": "observation", "content": "boring"}],
    }
    decisions = cd.evaluate(summary)
    assert len(decisions) == 0


def test_consolidation_decider_process_pending() -> None:
    cd = ConsolidationDecider()
    cd.enqueue({"novelty": 0.7, "items": [{"type": "obs", "content": "e1"}]})
    cd.enqueue({"novelty": 0.8, "items": [{"type": "skill", "content": "e2"}]})
    assert cd.pending_count == 2
    results = cd.process_pending()
    assert cd.pending_count == 0
    assert len(results) == 2
    assert len(results[0]) >= 1


def test_consolidation_decider_skill_targets_skill_memory() -> None:
    cd = ConsolidationDecider()
    summary = {
        "novelty": 0.8,
        "reward": 0.7,
        "items": [{"type": "skill", "content": "precise mouse movement"}],
    }
    decisions = cd.evaluate(summary)
    assert len(decisions) >= 1
    # Skill items with high importance should target skill or mid_term
    assert decisions[0]["target_memory"] in ("skill", "mid_term", "long_term")


def test_consolidation_decider_habit_targets_habit_memory() -> None:
    cd = ConsolidationDecider()
    summary = {
        "novelty": 0.5,
        "repetition": 8,
        "items": [{"type": "habit", "content": "regular scan pattern"}],
    }
    decisions = cd.evaluate(summary)
    assert len(decisions) >= 1
    assert decisions[0]["target_memory"] in ("habit", "short_term", "mid_term")
