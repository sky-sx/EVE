"""
Searchable episode log explorer with query and filter support — recent
episodes, episode statistics, trace summary.
"""

from dataclasses import dataclass


@dataclass
class LogExplorer:
    """Searchable episode log explorer with query and filter support.

    Renders episode log summaries and statistics with search and date-range
    filtering capabilities.
    """

    def render(self, state: dict) -> str:
        """Render episode log panel.

        Args:
            state: Dict with keys for episodes, episode_stats,
                   trace_summary.

        Returns:
            Formatted searchable log display with episode filtering capabilities.
        """
        ...

    def search(self, query: str) -> str:
        """Search logs for episodes matching the given query.

        Args:
            query: Search string to match against episode logs.

        Returns:
            Formatted search results string.
        """
        ...

    def filter_by_date(self, start: float, end: float) -> str:
        """Filter logs to episodes within a date range.

        Args:
            start: Start timestamp (inclusive).
            end: End timestamp (inclusive).

        Returns:
            Formatted filtered log display string.
        """
        ...
