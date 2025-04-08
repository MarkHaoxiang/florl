from pathlib import Path


class StatefulClient:
    """Mixin for clients to save and load state from disk."""

    def __init__(self, root_dir: Path, node_id: int):
        """Initialize StatefulClient.

        Args:
            root_dir (Path): Root directory for storing client state.
            node_id (int): Unique identifier for the client node.
        """
        super().__init__()
        self._working_dir = Path(root_dir, "state", f"{node_id}")
        self._working_dir.mkdir(parents=True, exist_ok=True)

    @property
    def working_dir(self) -> Path:
        """Returns the working directory for this client.

        Returns:
            Path: Path to the directory where client-specific state is stored.
        """
        return self._working_dir
