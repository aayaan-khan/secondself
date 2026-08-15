import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import capture


class TestCapturePipeline(unittest.TestCase):
    def setUp(self):
        # Use a temporary directory for raw captures during testing
        self.test_dir = tempfile.mkdtemp()
        self.raw_test_dir = Path(self.test_dir) / "raw"
        self.raw_test_dir.mkdir(parents=True, exist_ok=True)
        os.environ["SECONDSELF_RAW_DIR"] = str(self.raw_test_dir)
        self.orig_raw_dir = capture.RAW_DIR
        capture.RAW_DIR = self.raw_test_dir

    def tearDown(self):
        capture.RAW_DIR = self.orig_raw_dir
        if "SECONDSELF_RAW_DIR" in os.environ:
            del os.environ["SECONDSELF_RAW_DIR"]
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_capture_note_success(self):
        res = capture.capture_item("note", "My test note content")
        self.assertEqual(res["status"], "saved")
        self.assertEqual(res["type"], "note")
        
        file_path = Path(res["path"])
        self.assertTrue(file_path.exists())
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.assertEqual(data["id"], res["id"])
        self.assertEqual(data["type"], "note")
        self.assertEqual(data["content"], "My test note content")
        self.assertIsNone(data["source_path"])
        self.assertIn("timestamp", data)

    def test_capture_empty_note_rejected(self):
        with self.assertRaises(ValueError):
            capture.capture_item("note", "")
        with self.assertRaises(ValueError):
            capture.capture_item("note", "    \n   ")

    def test_capture_url_fallback_on_dead_link(self):
        dead_url = "https://this-domain-does-not-exist-secondself-test.org/abc"
        res = capture.capture_item("link", dead_url)
        self.assertEqual(res["status"], "saved")
        self.assertEqual(res["type"], "link")
        
        file_path = Path(res["path"])
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["content"], dead_url)
        self.assertIsNone(data["source_path"])

    def test_capture_pdf_file(self):
        import pypdf
        writer = pypdf.PdfWriter()
        page = writer.add_blank_page(width=200, height=200)
        # Write PDF to file
        pdf_file = Path(self.test_dir) / "document.pdf"
        with open(pdf_file, "wb") as f:
            writer.write(f)

        res = capture.capture_item("file", str(pdf_file))
        self.assertEqual(res["status"], "saved")
        self.assertEqual(res["type"], "file")
        self.assertEqual(res["record"]["source_path"], str(pdf_file.resolve()))

    def test_capture_text_file(self):
        sample_file = Path(self.test_dir) / "notes.txt"
        sample_file.write_text("Detailed notes from design discussion.\nSecond line.", encoding="utf-8")
        
        res = capture.capture_item("file", str(sample_file))
        self.assertEqual(res["status"], "saved")
        self.assertEqual(res["type"], "file")
        
        file_path = Path(res["path"])
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("Detailed notes from design discussion.", data["content"])
        self.assertEqual(data["source_path"], str(sample_file.resolve()))

    def test_capture_binary_file_stores_null_content(self):
        bin_file = Path(self.test_dir) / "archive.zip"
        bin_file.write_bytes(b"PK\x03\x04fakearchivedata")
        
        res = capture.capture_item("file", str(bin_file))
        self.assertEqual(res["status"], "saved")
        self.assertEqual(res["type"], "file")
        
        file_path = Path(res["path"])
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsNone(data["content"])
        self.assertEqual(data["source_path"], str(bin_file.resolve()))

    def test_large_file_truncation(self):
        large_file = Path(self.test_dir) / "huge.txt"
        large_content = "A" * 60000
        large_file.write_text(large_content, encoding="utf-8")
        
        res = capture.capture_item("file", str(large_file))
        file_path = Path(res["path"])
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(data["content"].endswith("[...truncated...]"))
        self.assertLess(len(data["content"]), 51000)

    def test_duplicate_detection(self):
        res1 = capture.capture_item("note", "Unique idea")
        self.assertEqual(res1["status"], "saved")
        
        # Second identical capture without force
        res2 = capture.capture_item("note", "Unique idea", force=False)
        self.assertEqual(res2["status"], "duplicate")
        self.assertEqual(res2["id"], res1["id"])
        
        # Third identical capture WITH force
        res3 = capture.capture_item("note", "Unique idea", force=True)
        self.assertEqual(res3["status"], "saved")
        self.assertNotEqual(res3["id"], res1["id"])

    def test_cli_execution(self):
        python_bin = sys.executable
        cli_script = PROJECT_ROOT / "capture.py"
        
        # Test CLI note capture
        result = subprocess.run(
            [python_bin, str(cli_script), "CLI capture test"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(PROJECT_ROOT)
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Captured [NOTE]", result.stdout)
        
        # Test CLI empty rejection
        result_empty = subprocess.run(
            [python_bin, str(cli_script), ""],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(PROJECT_ROOT)
        )
        self.assertNotEqual(result_empty.returncode, 0)


if __name__ == "__main__":
    unittest.main()
