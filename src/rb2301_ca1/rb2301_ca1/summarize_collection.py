"""Summarize expert/DAgger collection diagnostics from all worker shards."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_events(run_dir: Path) -> list[dict]:
    paths = sorted((run_dir / "collection_logs").glob("worker_*.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"No collection_logs/worker_*.jsonl files found under {run_dir}"
        )
    events: list[dict] = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
    if not events:
        raise ValueError(f"Collection logs under {run_dir} are empty")
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--show-failures",
        type=int,
        default=10,
        help="Number of earliest non-success episodes to print",
    )
    args = parser.parse_args()
    if args.show_failures < 0:
        raise ValueError("--show-failures cannot be negative")

    events = load_events(args.run_dir)
    outcomes = Counter(str(event.get("reason", "unknown")) for event in events)
    successes = outcomes.get("success", 0)
    retained = sum(bool(event.get("retained")) for event in events)
    steps = np.asarray([event.get("steps", 0) for event in events], dtype=float)
    minimum_scans = np.asarray(
        [event.get("minimum_scan", np.nan) for event in events],
        dtype=float,
    )
    print(
        f"attempted={len(events)} retained={retained} successes={successes} "
        f"success_rate={successes / len(events):.1%}"
    )
    print(
        "outcomes="
        + " ".join(
            f"{reason}:{count}"
            for reason, count in sorted(outcomes.items())
        )
    )
    print(
        f"steps_mean={np.mean(steps):.1f} steps_median={np.median(steps):.1f} "
        f"minimum_scan_mean={np.nanmean(minimum_scans):.3f}"
    )

    failures = [event for event in events if event.get("reason") != "success"]
    for event in failures[: args.show_failures]:
        print(
            f"failure worker={int(event.get('worker', -1)):02d} "
            f"attempt={int(event.get('attempt', -1)):04d} "
            f"layout_seed={event.get('layout_seed')} "
            f"reason={event.get('reason')} steps={event.get('steps')} "
            f"minimum_scan={float(event.get('minimum_scan', np.nan)):.3f} "
            f"final=({event.get('final_x')}, {event.get('final_y')}) "
            f"requested={event.get('final_requested_action')} "
            f"applied={event.get('final_applied_action')}"
        )


if __name__ == "__main__":
    main()
