"""
Model Registry — versioned model/policy weight storage.

Maintains versioned snapshots of policy/model weights as JSON files
on disk under runs/registry/. Cold-path only.
"""

from dataclasses import dataclass, field


@dataclass
class ModelRegistry:
    """Registry for versioned model/policy weights.

    Stores each registered version as a JSON file and tracks the
    current production version via a pointer file.

    All I/O is synchronous, file-backed, and cold-path only.
    """

    def register(self, name: str, weights: dict, metadata: dict | None = None) -> str:
        """Register a new version of a model.

        Args:
            name: Model identifier (e.g. 'visual_servo_policy').
            weights: Dictionary of parameter names to weight arrays.
                Arrays are converted to lists for JSON serialization.
            metadata: Optional dict with extra info (source, timestamp, etc.).

        Returns:
            version_id: Unique version string (UUID4 hex).
        """
        ...

    def get(self, name: str, version: str) -> dict:
        """Retrieve a specific version of a model.

        Args:
            name: Model identifier.
            version: Version string as returned by register().

        Returns:
            Dict with keys 'name', 'version', 'weights', 'metadata'.
            Weights values are lists (as stored).

        Raises:
            FileNotFoundError: If the version does not exist.
        """
        ...

    def list_versions(self, name: str) -> list[str]:
        """List all version IDs for a given model name.

        Args:
            name: Model identifier.

        Returns:
            Sorted list of version strings, newest first.
        """
        ...

    def promote(self, name: str, version: str) -> None:
        """Mark a specific version as the production version.

        Args:
            name: Model identifier.
            version: Version string to promote.

        Raises:
            FileNotFoundError: If the version does not exist.
        """
        ...

    def get_production(self, name: str) -> dict:
        """Retrieve the current production version of a model.

        Args:
            name: Model identifier.

        Returns:
            Dict with keys 'name', 'version', 'weights', 'metadata',
            or an empty dict if no production version is set.
        """
        ...
