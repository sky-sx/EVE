"""Memory 检索：关键词 + 时间 + 类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eve.memory.memorizer import Memorizer
    from eve.memory.indexes import IndexManager


@dataclass
class RetrievalRequest:
    keywords: list[str] | None = None
    payload_types: list[str] | None = None
    time_start_ns: int | None = None
    time_end_ns: int | None = None
    top_k: int = 10
    from_mtm_only: bool = False


@dataclass
class RetrievalResult:
    memory_id: str
    match_info: str            # Why this was matched
    score: float = 1.0


class Retriever:
    """简单关键词 + 时间范围 + 类型过滤检索器。"""

    def __init__(self, memorizer: Memorizer, index_manager: IndexManager) -> None:
        self._memorizer = memorizer
        self._index_manager = index_manager

    def search(self, request: RetrievalRequest) -> list[RetrievalResult]:
        results: dict[str, RetrievalResult] = {}

        # Determine candidate set
        if request.from_mtm_only:
            candidate_ids = self._memorizer.get_mtm_ids()
        else:
            candidate_ids = list(self._memorizer.catalog._entries.keys())

        # Filter by payload type
        if request.payload_types:
            candidate_ids = [
                mid for mid in candidate_ids
                if self._memorizer.catalog.lookup(mid) is not None
                and self._memorizer.catalog.lookup(mid).payload_type in request.payload_types  # type: ignore[union-attr]
            ]

        # Filter by time range
        if request.time_start_ns is not None or request.time_end_ns is not None:
            filtered: list[str] = []
            for mid in candidate_ids:
                entry = self._memorizer.catalog.lookup(mid)
                if entry is None:
                    continue
                if request.time_start_ns is not None and entry.created_at_ns < request.time_start_ns:
                    continue
                if request.time_end_ns is not None and entry.created_at_ns > request.time_end_ns:
                    continue
                filtered.append(mid)
            candidate_ids = filtered

        # Keyword matching
        if request.keywords:
            for mid in candidate_ids:
                best_score = 0.0
                matched_kw: list[str] = []
                payload = self._memorizer.read(mid)
                if payload is None:
                    continue
                text = self._payload_to_text(payload)
                if not text:
                    continue
                text_lower = text.lower()
                for kw in request.keywords:
                    kw_lower = kw.lower()
                    if kw_lower in text_lower:
                        matched_kw.append(kw)
                        best_score = max(best_score, 1.0)
                if matched_kw:
                    results[mid] = RetrievalResult(
                        memory_id=mid,
                        match_info=f"keywords matched: {', '.join(matched_kw)}",
                        score=best_score,
                    )
        else:
            # No keywords: all filtered candidates are results
            for mid in candidate_ids:
                results[mid] = RetrievalResult(
                    memory_id=mid,
                    match_info="type/time filter match",
                    score=1.0,
                )

        # Sort by score descending, then by time descending, take top_k
        sorted_results = sorted(
            results.values(),
            key=lambda r: (
                r.score,
                self._memorizer.catalog.lookup(r.memory_id).created_at_ns  # type: ignore[union-attr]
                if self._memorizer.catalog.lookup(r.memory_id) else 0,
            ),
            reverse=True,
        )
        return sorted_results[:request.top_k]

    def keyword_search(self, keyword: str, top_k: int = 10) -> list[str]:
        """Simple keyword match across text and json payloads."""
        request = RetrievalRequest(
            keywords=[keyword],
            payload_types=["text", "json"],
            top_k=top_k,
        )
        results = self.search(request)
        return [r.memory_id for r in results]

    @staticmethod
    def _payload_to_text(payload: object) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            try:
                import json
                return json.dumps(payload, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(payload)
        if isinstance(payload, list):
            return " ".join(str(item) for item in payload)
        return str(payload) if payload is not None else ""
