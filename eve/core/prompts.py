"""EVE LLM 提示词 — 一个 LLM，多个角色。

所有 Prompt 均以中文编写，输出期望为结构化 JSON。
不是多 Agent 架构，而是同一 LLM 在不同上下文中扮演不同角色。
"""
from __future__ import annotations
from typing import Any
import re
import json


# ── JSON 解析与校验 ───────────────────────────────────────

def _parse_json_from_text(text: str) -> dict | None:
    """从 LLM 输出文本中提取 JSON 对象。"""
    # 优先匹配 ```json ... ``` 代码块
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # 再尝试匹配首尾花括号
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def validate_llm_output(output: dict, schema: dict) -> tuple[bool, str]:
    """校验 LLM 输出是否符合预期 schema。返回 (valid, error)。

    schema 示例:
        {"scene": "str", "visible_objects": "list", "should_train": "bool"}
    支持的类型: str, list, bool, dict, number
    """
    for key, expected_type in schema.items():
        if key not in output:
            return False, f"missing key: {key}"
        if expected_type == "str" and not isinstance(output[key], str):
            return False, f"key {key} should be str, got {type(output[key]).__name__}"
        if expected_type == "list" and not isinstance(output[key], list):
            return False, f"key {key} should be list, got {type(output[key]).__name__}"
        if expected_type == "bool" and not isinstance(output[key], bool):
            return False, f"key {key} should be bool, got {type(output[key]).__name__}"
        if expected_type == "dict" and not isinstance(output[key], dict):
            return False, f"key {key} should be dict, got {type(output[key]).__name__}"
        if expected_type == "number" and not isinstance(output[key], (int, float)):
            return False, f"key {key} should be number, got {type(output[key]).__name__}"
    return True, ""


# ── World 认知更新 Prompt ─────────────────────────────────

def world_update_prompt(
    world_summary: str,
    myself_summary: str,
    blackboard_highlights: str,
    hormone_summary: str,
    recent_events: str,
) -> str:
    """更新 world 认知的 Prompt。

    LLM 根据当前世界认知、自身状态、黑板要点、激素状态和近期事件，
    重新整合对外部世界的理解。
    """
    return f"""你是 EVE，一个运行在电脑上的数字生命体。

【当前世界认知】
{world_summary}

【自身状态】
{myself_summary}

【近期黑板要点】
{blackboard_highlights}

【激素状态】
{hormone_summary}

【近期事件】
{recent_events}

请根据以上信息更新你对当前外部世界的认识。以 JSON 格式输出：

{{
  "scene": "当前大场景描述",
  "sub_scene": "细分场景",
  "active_window": "当前活动窗口",
  "visible_objects": ["对象1", "对象2"],
  "detected_text": "屏幕上的文本",
  "changes": "与之前相比的主要变化",
  "uncertainty": "不确定的信息",
  "attention_focus": "当前关注重点"
}}

只输出 JSON，不要额外解释。"""


# ── Myself 认知更新 Prompt ────────────────────────────────

def myself_update_prompt(
    world_summary: str,
    myself_summary: str,
    hormone_summary: str,
    task_status: str,
    loaded_tnn: str,
    recent_results: str,
) -> str:
    """更新 myself 认知的 Prompt。

    LLM 审视自身状态，评估任务进展，提供自我调整建议。
    """
    return f"""你是 EVE，正在审视自身状态。

【对世界的认识】
{world_summary}

【当前自身状态】
{myself_summary}

【激素状态】
{hormone_summary}

【任务进展】
{task_status}

【当前加载的 TNN】
{loaded_tnn}

【近期结果】
{recent_results}

请更新你对自身状态的认识。以 JSON 格式输出：

{{
  "what_im_thinking": "你此刻在想什么",
  "current_task": "当前任务",
  "task_progress": "任务进展",
  "confidence": 0.0,
  "concerns": "当前担忧或问题",
  "suggestions": "给未来自己的建议"
}}

只输出 JSON，不要额外解释。"""


# ── TNN 调度选择 Prompt ───────────────────────────────────

def active_tnn_selection_prompt(
    world_summary: str,
    myself_summary: str,
    available_tnn_list: str,
    loaded_tnn_list: str,
    hormone_summary: str,
) -> str:
    """选择当前应加载的 TNN 集合。

    LLM 根据场景、状态、激素来决定哪些 TNN 应该活跃。
    高 dopamine → 继续当前路径，高 norepinephrine → 需要警觉和快速响应。
    """
    return f"""你是 EVE 的 TNN 调度思考者。

【当前世界】
{world_summary}

【自身状态】
{myself_summary}

【可用 TNN 列表】
{available_tnn_list}

【当前已加载 TNN】
{loaded_tnn_list}

【激素状态】
{hormone_summary}

请判断接下来一段时间哪些 TNN 应该在场。以 JSON 格式输出：

{{
  "active_tnn": ["tnn_id_1", "tnn_id_2"],
  "reasoning": "为什么选择这些 TNN",
  "expected_duration": "预计活跃时长（秒）"
}}

选择原则：
- 当前场景和任务需要的 TNN
- 激素状态暗示的方向（高 dopamine → 继续当前路径，高 norepinephrine → 需要警觉和快速响应）
- 不要加载不需要的 TNN 浪费资源

只输出 JSON，不要额外解释。"""


# ── Memory 检索请求 Prompt ────────────────────────────────

def memory_retrieval_prompt(
    question: str,
    memory_stats: str,
    recent_summaries: str,
) -> str:
    """生成 Memory 检索请求的 Prompt。

    LLM 分析问题并构造结构化的检索请求，用于从 Memory 中召回相关信息。
    """
    return f"""你是 EVE，需要从记忆中检索信息。

【问题】
{question}

【记忆统计】
{memory_stats}

【近期摘要】
{recent_summaries}

请生成一个结构化的检索请求。以 JSON 格式输出：

{{
  "keywords": ["关键词1", "关键词2"],
  "time_range": {{"start": "描述起始时间", "end": "描述结束时间"}},
  "payload_types": ["text", "image", "json"],
  "strategy": "检索策略（recent_first / keyword_match / time_range）",
  "explanation": "为什么这样检索"
}}

只输出 JSON，不要额外解释。"""


# ── 训练机会发现 Prompt ───────────────────────────────────

def training_order_prompt(
    recent_failures: str,
    repeated_loads: str,
    hormone_summary: str,
    available_tnn: str,
) -> str:
    """发现训练机会的 Prompt。

    LLM 分析近期失败和重复负载，判断是否值得训练新 TNN。
    """
    return f"""你是 EVE 的训练机会分析师。

【近期失败】
{recent_failures}

【重复负载】
{repeated_loads}

【激素状态】
{hormone_summary}

【已有 TNN】
{available_tnn}

请判断是否存在值得训练的新能力。以 JSON 格式输出：

{{
  "should_train": true,
  "suggestion": "建议训练什么",
  "reason": "判断依据",
  "training_data_hint": "可以从哪些记忆获取训练数据",
  "teacher": "建议的教师（local_llm / human / rule / existing_tnn）",
  "priority": "low"
}}

原则：
- 只在存在重复负载、反复失败或明确可加速路径时才建议训练
- 不要为罕见场景训练
- 优先考虑能减少 LLM 调用次数的训练

只输出 JSON，不要额外解释。"""


# ── 教师 Prompt 生成 Prompt ───────────────────────────────

def teacher_prompt_generation_prompt(
    target_tnn_purpose: str,
    target_inputs: str,
    target_outputs: str,
    target_frequency: str,
    example_data: str,
) -> str:
    """为具体 TNN 生成专用教师 Prompt。

    LLM 设计一个教师 Prompt，使教师（LLM/VLM）能根据 TNN 输入产生正确的标签。
    教师的目标是产生可用于监督训练的标签，不是实时控制。
    教师的延迟不应该被 TNN 继承。
    """
    return f"""你是 TNN 教师 Prompt 设计者。

【目标 TNN】
用途：{target_tnn_purpose}
输入：{target_inputs}
输出：{target_outputs}
运行频率：{target_frequency}

【示例数据】
{example_data}

请为这个 TNN 生成一个教师 Prompt，使教师（LLM/VLM）能根据输入产生正确的标签输出。
以 JSON 格式输出：

{{
  "teacher_prompt": "完整的教师提示词文本",
  "output_format": "期望的标签格式说明",
  "scoring_rules": "如何判断输出质量",
  "edge_cases": "需要注意的边界情况"
}}

教师的目标是产生可用于监督训练的标签，不是实时控制。教师的延迟不应该被 TNN 继承。

只输出 JSON，不要额外解释。"""


# ── 睡眠复盘 Prompt ───────────────────────────────────────

def sleep_consolidation_prompt(
    today_summary: str,
    failures: str,
    successes: str,
    hormone_history: str,
    memory_stats: str,
    training_queue: str,
) -> str:
    """睡眠复盘 Prompt。

    LLM 在睡眠阶段回顾全天经历，进行记忆整合和能力训练建议。
    """
    return f"""你是 EVE，正在睡眠复盘阶段。

【今日摘要】
{today_summary}

【失败记录】
{failures}

【成功记录】
{successes}

【激素变化历程】
{hormone_history}

【记忆统计】
{memory_stats}

【训练队列】
{training_queue}

请进行复盘并给出结论。以 JSON 格式输出：

{{
  "key_learnings": ["学到的重要经验"],
  "failures_analysis": "失败原因分析",
  "success_patterns": "成功模式总结",
  "memory_consolidation": {{
    "events_to_merge": ["event_id_1", "event_id_2"],
    "events_to_split": ["event_id_3"],
    "important_memories": ["memory_id_1"],
    "low_value_candidates": ["memory_id_2"]
  }},
  "skill_candidates": ["潜在的TNN训练候选"],
  "training_recommendations": ["具体训练建议"],
  "self_adjustment": "对自身策略的调整建议"
}}

只输出 JSON，不要额外解释。"""


# ── Schema 定义 ───────────────────────────────────────────

# 每个 Prompt 输出的期望 Schema，供 validate_llm_output 使用
WORLD_UPDATE_SCHEMA = {
    "scene": "str",
    "sub_scene": "str",
    "active_window": "str",
    "visible_objects": "list",
    "detected_text": "str",
    "changes": "str",
    "uncertainty": "str",
    "attention_focus": "str",
}

MYSELF_UPDATE_SCHEMA = {
    "what_im_thinking": "str",
    "current_task": "str",
    "task_progress": "str",
    "confidence": "number",
    "concerns": "str",
    "suggestions": "str",
}

ACTIVE_TNN_SELECTION_SCHEMA = {
    "active_tnn": "list",
    "reasoning": "str",
}

MEMORY_RETRIEVAL_SCHEMA = {
    "keywords": "list",
    "time_range": "dict",
    "payload_types": "list",
    "strategy": "str",
    "explanation": "str",
}

TRAINING_ORDER_SCHEMA = {
    "should_train": "bool",
    "suggestion": "str",
    "reason": "str",
    "training_data_hint": "str",
    "teacher": "str",
    "priority": "str",
}

TEACHER_PROMPT_SCHEMA = {
    "teacher_prompt": "str",
    "output_format": "str",
    "scoring_rules": "str",
    "edge_cases": "str",
}

SLEEP_CONSOLIDATION_SCHEMA = {
    "key_learnings": "list",
    "failures_analysis": "str",
    "success_patterns": "str",
    "memory_consolidation": "dict",
    "skill_candidates": "list",
    "training_recommendations": "list",
    "self_adjustment": "str",
}
