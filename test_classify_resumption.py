import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import classify


class TestClassifyResumption(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = Path(self.test_dir) / "raw"
        self.wiki_dir = Path(self.test_dir) / "wiki"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.wiki_dir.mkdir(parents=True, exist_ok=True)

        os.environ["SECONDSELF_RAW_DIR"] = str(self.raw_dir)
        os.environ["SECONDSELF_WIKI_DIR"] = str(self.wiki_dir)
        self.orig_raw = classify.RAW_DIR
        self.orig_wiki = classify.WIKI_DIR
        classify.RAW_DIR = self.raw_dir
        classify.WIKI_DIR = self.wiki_dir

    def tearDown(self):
        classify.RAW_DIR = self.orig_raw
        classify.WIKI_DIR = self.orig_wiki
        if "SECONDSELF_RAW_DIR" in os.environ:
            del os.environ["SECONDSELF_RAW_DIR"]
        if "SECONDSELF_WIKI_DIR" in os.environ:
            del os.environ["SECONDSELF_WIKI_DIR"]
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_raw_item(self, item_id: str, content: str):
        file_path = self.raw_dir / f"2026-08-19_{item_id}.json"
        data = {
            "id": item_id,
            "type": "note",
            "timestamp": "2026-08-19T10:00:00Z",
            "content": content,
            "source_path": None
        }
        file_path.write_text(json.dumps(data), encoding="utf-8")
        return file_path

    @patch("classify.get_groq_client")
    def test_mid_batch_crash_and_resumption(self, mock_get_client):
        # Create 3 raw items
        self._create_raw_item("item-1", "First item")
        self._create_raw_item("item-2", "Second item")
        self._create_raw_item("item-3", "Third item")

        # Mock LLM client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "category": "Resources",
            "tags": ["test"],
            "summary": "Summary test"
        })
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        # Simulate batch classify that crashes on 3rd item
        call_count = [0]
        def fake_classify_raw(path, client=None, force=False, wiki_dir=None):
            call_count[0] += 1
            if "item-3" in path.name and call_count[0] == 3:
                raise KeyboardInterrupt("Simulated mid-batch interrupt/crash on item 3")
            return orig_classify_raw_file(path, client=client, force=force, wiki_dir=wiki_dir)

        orig_classify_raw_file = classify.classify_raw_file
        with patch("classify.classify_raw_file", side_effect=fake_classify_raw):
            try:
                classify.batch_classify(wiki_dir=self.wiki_dir)
            except KeyboardInterrupt:
                pass

        # Verify: item-1 and item-2 notes were created in wiki/, item-3 was not
        self.assertTrue((self.wiki_dir / "Resources" / "item-1.md").exists())
        self.assertTrue((self.wiki_dir / "Resources" / "item-2.md").exists())
        self.assertFalse((self.wiki_dir / "Resources" / "item-3.md").exists())

        # Reset call counts
        mock_client.chat.completions.create.reset_mock()

        # Run resumption: classify.batch_classify() should classify item-3 and SKIP item-1 & item-2
        results = classify.batch_classify(wiki_dir=self.wiki_dir)

        # Confirm results
        statuses = {r["id"]: r["status"] for r in results if "id" in r}
        self.assertEqual(statuses["item-1"], "skipped")
        self.assertEqual(statuses["item-2"], "skipped")
        self.assertEqual(statuses["item-3"], "classified")

        # Confirm only ONE LLM API call was made during resumption (for item-3)
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)

        # Confirm all 3 files now exist in wiki/
        self.assertTrue((self.wiki_dir / "Resources" / "item-3.md").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
