import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recording_tools import (
    build_transcription_definition,
    discover_uploaded_recording,
    resolve_session_file,
    save_meeting_report,
)


class ResolveSessionFileTests(unittest.TestCase):
    def test_uses_multilingual_mode_by_default(self):
        definition = build_transcription_definition("", 10)

        self.assertNotIn("locales", definition)
        self.assertEqual(definition["diarization"]["maxSpeakers"], 10)

    def test_uses_fixed_locale_when_explicitly_requested(self):
        definition = build_transcription_definition(" fr-FR ", 6)

        self.assertEqual(definition["locales"], ["fr-FR"])
        self.assertEqual(definition["diarization"]["maxSpeakers"], 6)

    def test_accepts_supported_file_under_home(self):
        home = Path.home()
        with tempfile.TemporaryDirectory(dir=home) as directory:
            path = Path(directory) / "meeting.mp4"
            path.write_bytes(b"synthetic")

            self.assertEqual(resolve_session_file(str(path)), path.resolve())

    def test_resolves_relative_file_from_home(self):
        home = Path.home()
        with tempfile.TemporaryDirectory(dir=home) as directory:
            path = Path(directory) / "meeting.mp4"
            path.write_bytes(b"synthetic")
            relative_path = path.relative_to(home)

            self.assertEqual(resolve_session_file(str(relative_path)), path.resolve())

    def test_resolves_virtual_absolute_path_from_home(self):
        home = Path.home()
        virtual_directory = home / "data"
        virtual_directory.mkdir(exist_ok=True)
        path = virtual_directory / "virtual-meeting.mp4"
        path.write_bytes(b"synthetic")
        try:
            self.assertEqual(
                resolve_session_file("/data/virtual-meeting.mp4"),
                path.resolve(),
            )
        finally:
            path.unlink(missing_ok=True)

    def test_rejects_file_outside_home(self):
        outside_home = Path(os.environ.get("SystemRoot", "C:\\Windows")) / "win.ini"
        if not outside_home.exists():
            self.skipTest("No stable file outside the user home is available")

        with self.assertRaisesRegex(ValueError, "session directory"):
            resolve_session_file(str(outside_home))

    def test_rejects_unsupported_extension(self):
        home = Path.home()
        with tempfile.TemporaryDirectory(dir=home) as directory:
            path = Path(directory) / "meeting.exe"
            path.write_bytes(b"synthetic")

            with self.assertRaisesRegex(ValueError, "Unsupported meeting file type"):
                resolve_session_file(str(path))

    def test_discovers_only_uploaded_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "meeting.mp4"
            path.write_bytes(b"synthetic")

            with patch("recording_tools._session_roots", return_value=[root]):
                self.assertEqual(discover_uploaded_recording(), path.resolve())

    def test_requires_filename_when_multiple_recordings_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first.mp4").write_bytes(b"synthetic")
            (root / "second.wav").write_bytes(b"synthetic")

            with patch("recording_tools._session_roots", return_value=[root]):
                with self.assertRaisesRegex(ValueError, "Multiple uploaded recordings"):
                    discover_uploaded_recording()

    def test_saves_downloadable_summary_beside_transcript(self):
        home = Path.home()
        with tempfile.TemporaryDirectory(dir=home) as directory:
            transcript = Path(directory) / "meeting.mp4.transcript.json"
            transcript.write_text("{}", encoding="utf-8")

            result = save_meeting_report("# Summary", str(transcript))

            self.assertIn("meeting.mp4.summary.md", result)
            self.assertEqual(
                (Path(directory) / "meeting.mp4.summary.md").read_text(
                    encoding="utf-8"
                ),
                "# Summary\n",
            )


if __name__ == "__main__":
    unittest.main()
