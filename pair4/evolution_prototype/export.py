"""Versioned, portable exports. Timing is separate from deterministic histories."""

import json
from pathlib import Path

from .search import SearchResult


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_run(result: SearchResult, directory: str | Path) -> Path:
    """Create a new output directory, refusing to overwrite an existing run."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "manifest.json", result.manifest())
    with (directory / "history.jsonl").open("w", encoding="utf-8") as stream:
        for event in result.events:
            stream.write(json.dumps(event.to_dict(), sort_keys=True, allow_nan=False) + "\n")
    write_json(directory / "summary.json", result.summary())
    return directory
