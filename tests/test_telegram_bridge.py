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


if __name__ == "__main__":
    unittest.main()
