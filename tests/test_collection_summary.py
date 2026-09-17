import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rb2301_ca1.summarize_collection import load_events


class CollectionSummaryTests(unittest.TestCase):
    def test_events_from_all_worker_logs_are_loaded(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            log_dir = run_dir / "collection_logs"
            log_dir.mkdir()
            (log_dir / "worker_00.jsonl").write_text(
                json.dumps({"worker": 0, "reason": "success"}) + "\n",
                encoding="utf-8",
            )
            (log_dir / "worker_01.jsonl").write_text(
                json.dumps({"worker": 1, "reason": "collision"}) + "\n",
                encoding="utf-8",
            )
            events = load_events(run_dir)
            self.assertEqual(len(events), 2)
            self.assertEqual(
                {event["reason"] for event in events},
                {"success", "collision"},
            )


if __name__ == "__main__":
    unittest.main()
