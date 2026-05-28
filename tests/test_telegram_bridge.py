"""Unit tests for telegram_bridge.py — pure functions, no network."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import telegram_bridge as tb  # noqa: E402


class TestLoadSecrets(unittest.TestCase):
    def test_parses_valid_secrets(self):
        with tempfile.NamedTemporaryFile("w", suffix=".secrets", delete=False) as f:
            f.write("# comment line\n")
            f.write("TELEGRAM_BOT_TOKEN=abc:123\n")
            f.write("\n")
            f.write("TELEGRAM_CHAT_ID=987654\n")
            path = Path(f.name)
        try:
            token, chat = tb.load_secrets(path)
            self.assertEqual(token, "abc:123")
            self.assertEqual(chat, 987654)
        finally:
            path.unlink()

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            tb.load_secrets(Path("/nonexistent/secrets"))

    def test_missing_token_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".secrets", delete=False) as f:
            f.write("TELEGRAM_CHAT_ID=1\n")
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError):
                tb.load_secrets(path)
        finally:
            path.unlink()


class TestReaders(unittest.TestCase):
    def test_status_text_missing_file(self):
        out = tb.status_text(Path("/nonexistent/status.json"))
        self.assertIn("no cascade running", out)

    def test_status_text_valid(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({
                "phase": "train", "seed": 5, "config": "feng_hw_A",
                "started": "2026-05-28T10:00:00",
                "seed_started": "2026-05-28T10:00:00",
                "pid": 12345,
                "seeds_total": 5, "seeds_done": 0,
            }, f)
            path = Path(f.name)
        try:
            out = tb.status_text(path)
            self.assertIn("phase=train", out)
            self.assertIn("seed=5", out)
            self.assertIn("config=feng_hw_A", out)
        finally:
            path.unlink()

    def test_tail_text_no_logs(self):
        with tempfile.TemporaryDirectory() as d:
            out = tb.tail_text(Path(d), pattern="campaign_*.log", n=10)
            self.assertIn("no log", out)

    def test_tail_text_latest(self):
        with tempfile.TemporaryDirectory() as d:
            logp = Path(d) / "campaign_feng_20260528_120000.log"
            logp.write_text("line1\nline2\nline3\nline4\nline5\n")
            out = tb.tail_text(Path(d), pattern="campaign_*.log", n=3)
            self.assertIn("line3", out)
            self.assertIn("line5", out)
            self.assertNotIn("line1", out)

    def test_seeds_text_empty(self):
        with tempfile.TemporaryDirectory() as d:
            out = tb.seeds_text(Path(d), config="feng_hw_A")
            self.assertIn("no seeds", out)

    def test_seeds_text_with_summary(self):
        with tempfile.TemporaryDirectory() as d:
            seed_dir = Path(d) / "feng_hw_A" / "seed_5"
            seed_dir.mkdir(parents=True)
            (seed_dir / "eval_summary.csv").write_text(
                "maze,success_rate,avg_reward,avg_steps\n"
                "1,0.42,150.0,200\n"
                "2,0.55,180.0,210\n"
                "3,0.00,-500.0,80\n"
            )
            out = tb.seeds_text(Path(d), config="feng_hw_A")
            self.assertIn("seed_5", out)
            self.assertIn("done", out.lower())


class TestDispatcher(unittest.TestCase):
    def test_help(self):
        out = tb.handle("/help", config="feng_hw_A")
        self.assertIn("/status", out)
        self.assertIn("/tail", out)
        self.assertIn("/abort", out)

    def test_start_returns_help(self):
        out = tb.handle("/start", config="feng_hw_A")
        self.assertIn("/status", out)

    def test_unknown(self):
        out = tb.handle("/somethingweird", config="feng_hw_A")
        self.assertIn("unknown", out.lower())

    def test_empty(self):
        out = tb.handle("", config="feng_hw_A")
        self.assertIn("unknown", out.lower())

    def test_status_dispatch(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"phase": "test", "seed": 7, "config": "feng_hw_A"}, f)
            path = Path(f.name)
        try:
            out = tb.handle("/status", config="feng_hw_A", status_path=path)
            self.assertIn("phase=test", out)
        finally:
            path.unlink()

    def test_tail_with_n(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "campaign_x.log").write_text(
                "\n".join(f"row{i}" for i in range(50))
            )
            out = tb.handle("/tail 5", config="feng_hw_A", logs_dir=Path(d))
            self.assertIn("row49", out)
            self.assertIn("row45", out)
            self.assertNotIn("row44", out)

    def test_tail_default_n(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "campaign_x.log").write_text("\n".join(["x"] * 100))
            out = tb.handle("/tail", config="feng_hw_A", logs_dir=Path(d))
            self.assertIn("last 20 of 100", out)


class TestControl(unittest.TestCase):
    def test_pause_writes_control_file(self):
        with tempfile.TemporaryDirectory() as d:
            ctrl = Path(d) / "cascade_control"
            out = tb.handle("/pause", config="feng_hw_A", control_path=ctrl)
            self.assertIn("pause", out.lower())
            self.assertEqual(ctrl.read_text(), "pause")

    def test_resume_clears_control(self):
        with tempfile.TemporaryDirectory() as d:
            ctrl = Path(d) / "cascade_control"
            ctrl.write_text("pause")
            out = tb.handle("/resume", config="feng_hw_A", control_path=ctrl)
            self.assertIn("resume", out.lower())
            self.assertEqual(ctrl.read_text(), "")

    def test_abort_writes_abort_and_invokes_kill(self):
        called = {}

        def fake_kill():
            called["killed"] = True
            return True

        with tempfile.TemporaryDirectory() as d:
            ctrl = Path(d) / "cascade_control"
            out = tb.handle("/abort", config="feng_hw_A",
                            control_path=ctrl, kill_fn=fake_kill)
            self.assertIn("abort", out.lower())
            self.assertEqual(ctrl.read_text(), "abort")
            self.assertTrue(called.get("killed"))


if __name__ == "__main__":
    unittest.main()
