import json
import os
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recording_tools import (
    MAX_FILE_BYTES,
    MeetingProcessingError,
    _publish_artifacts,
    build_transcription_definition,
    discover_uploaded_recording,
    inspect_transcript,
    resolve_recording_path,
    save_meeting_report,
    transcribe_meeting_file,
)


HTTP_CLIENT = httpx.Client
TRANSCRIPT = json.dumps({
    "utterances": [
        {"speaker_id": "P1", "start_ms": 0, "end_ms": 1000, "text": "Pilot approved."}
    ]
})
SPEECH_RESPONSE = {
    "durationMilliseconds": 2200,
    "phrases": [
        {
            "speaker": 0, "offsetMilliseconds": 0, "durationMilliseconds": 1000,
            "text": "Pilot approved.", "locale": "en-US", "confidence": 0.9,
        },
        {
            "speaker": 1, "offsetMilliseconds": 1200, "durationMilliseconds": 1000,
            "text": "Bonjour.", "locale": "fr-FR",
        },
    ],
}


class SessionTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        env = patch.dict(os.environ, {
            "MEETING_SESSION_DIR": str(self.root),
            "AZURE_SPEECH_ENDPOINT": "https://speech.example.test",
        })
        env.start()
        self.addCleanup(env.stop)

    def recording(self, name="meeting.mp4", content=b"synthetic"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def transcript(self):
        path = self.root / "meeting.mp4.transcript.json"
        path.write_text(TRANSCRIPT, encoding="utf-8")
        return inspect_transcript(file_path=str(path))

    def speech_transport(self, status=200, payload=SPEECH_RESPONSE):
        self.requests = []

        def handler(request):
            request.read()
            self.requests.append(request)
            return httpx.Response(
                status, json=payload, headers={"apim-request-id": "test-request-123"}
            )

        transport = httpx.MockTransport(handler)
        client_patch = patch(
            "recording_tools.httpx.Client",
            side_effect=lambda **kwargs: HTTP_CLIENT(transport=transport, **kwargs),
        )
        client_patch.start()
        self.addCleanup(client_patch.stop)
        credential_patch = patch("recording_tools.DefaultAzureCredential")
        self.credential_factory = credential_patch.start()
        self.addCleanup(credential_patch.stop)
        credential = self.credential_factory.return_value.__enter__.return_value
        credential.get_token.return_value = SimpleNamespace(token="test-token")


class ResolveRecordingPathTests(SessionTestCase):
    def test_uses_multilingual_mode_by_default(self):
        definition = build_transcription_definition("", 10)
        self.assertNotIn("locales", definition)
        self.assertEqual(definition["diarization"]["maxSpeakers"], 10)
        self.assertEqual(definition["profanityFilterMode"], "Masked")

    def test_uses_fixed_locale_when_explicitly_requested(self):
        definition = build_transcription_definition(" fr-FR ", 6)
        self.assertEqual(definition["locales"], ["fr-FR"])
        self.assertEqual(definition["diarization"]["maxSpeakers"], 6)

    def test_rejects_invalid_speaker_counts_and_locale(self):
        for value in (True, 1, 36, 2.5, "5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_transcription_definition("", value)
        for value in (None, "French", "fr-FR,en-US"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_transcription_definition(value, 10)

    def test_accepts_absolute_and_relative_session_paths(self):
        path = self.recording("nested/meeting.mp4")
        for name in (str(path), str(path.relative_to(self.root))):
            with self.subTest(name=name):
                self.assertEqual(resolve_recording_path(name), path)

    def test_resolves_virtual_data_path_and_basename_consistently(self):
        path = self.recording("data/meeting.mp4")
        for name in ("", "meeting.mp4", "/data/meeting.mp4", " /data/meeting.mp4 ", str(path)):
            with self.subTest(name=name):
                self.assertEqual(resolve_recording_path(name), path)

    def test_disambiguates_nested_recordings_with_returned_paths(self):
        first = self.recording("data/first.mp4")
        self.recording("data/second.mp4")
        with self.assertRaisesRegex(ValueError, "Multiple uploaded recordings"):
            discover_uploaded_recording()
        self.assertEqual(resolve_recording_path("first.mp4"), first)

    def test_duplicate_basenames_require_relative_path(self):
        self.recording()
        nested = self.recording("data/meeting.mp4")
        with self.assertRaisesRegex(ValueError, "session-relative path"):
            resolve_recording_path("meeting.mp4")
        self.assertEqual(resolve_recording_path(str(nested.relative_to(self.root))), nested)

    def test_rejects_parent_traversal_and_external_absolute_path(self):
        for value in ("../meeting.mp4", str(self.root.parent / "external.mp4")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "session directory"):
                resolve_recording_path(value)

    def test_discovery_and_explicit_paths_reject_external_symlink_target(self):
        path = self.recording()
        original_resolve = Path.resolve

        def resolve(candidate, *args, **kwargs):
            if candidate == path:
                return self.root.parent / "external.mp4"
            return original_resolve(candidate, *args, **kwargs)

        with patch.object(Path, "resolve", resolve):
            for name in ("", "meeting.mp4"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "session directory"):
                    resolve_recording_path(name)

    def test_ignores_application_working_directory(self):
        with patch("recording_tools.Path.cwd", return_value=self.root.parent):
            with self.assertRaisesRegex(ValueError, "No matching uploaded"):
                discover_uploaded_recording()

    def test_rejects_external_data_directory_before_enumerating_it(self):
        data = self.root / "data"
        data.mkdir()
        original_resolve = Path.resolve
        original_iterdir = Path.iterdir

        def resolve(candidate, *args, **kwargs):
            if candidate == data:
                return self.root.parent / "external"
            return original_resolve(candidate, *args, **kwargs)

        def iterdir(candidate):
            if candidate == data:
                self.fail("An external session directory was enumerated.")
            return original_iterdir(candidate)

        with patch.object(Path, "resolve", resolve), patch.object(Path, "iterdir", iterdir):
            with self.assertRaisesRegex(ValueError, "session directory"):
                discover_uploaded_recording()

    def test_rejects_unsupported_extension(self):
        self.recording("meeting.exe")
        with self.assertRaisesRegex(ValueError, "Unsupported meeting file type"):
            resolve_recording_path("meeting.exe")

    def test_discovered_and_explicit_recordings_share_size_validation(self):
        path = self.recording()
        original_stat = Path.stat
        for size in (0, MAX_FILE_BYTES, MAX_FILE_BYTES + 1):
            def stat(candidate, *args, **kwargs):
                result = original_stat(candidate, *args, **kwargs)
                if candidate == path:
                    return SimpleNamespace(st_size=size, st_mode=result.st_mode)
                return result

            with patch.object(Path, "stat", stat):
                for name in ("", "meeting.mp4"):
                    with self.subTest(size=size, name=name), self.assertRaises(ValueError):
                        resolve_recording_path(name)

    def test_accepts_file_just_below_service_size_limit(self):
        path = self.recording()
        original_stat = Path.stat

        def stat(candidate, *args, **kwargs):
            result = original_stat(candidate, *args, **kwargs)
            if candidate == path:
                return SimpleNamespace(st_size=MAX_FILE_BYTES - 1, st_mode=result.st_mode)
            return result

        with patch.object(Path, "stat", stat):
            self.assertEqual(resolve_recording_path(""), path)


class TranscribeMeetingFileTests(SessionTestCase):
    def test_request_auth_multilingual_evidence_and_downloads(self):
        self.recording()
        self.speech_transport()
        result = transcribe_meeting_file()
        request = self.requests[0]
        self.assertEqual(request.headers["authorization"], "Bearer test-token")
        self.assertIn("api-version=2025-10-15", str(request.url))
        self.assertIn(b"video/mp4", request.content)
        self.assertIn(b'"enabled": true', request.content)
        self.assertNotIn(b'"locales"', request.content)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["meeting"]["speaker_count"], 2)
        self.assertEqual(result["meeting"]["detected_locale_counts"], {"en-US": 1, "fr-FR": 1})
        self.assertEqual(result["meeting"]["transcription_mode"], "multilingual")
        self.assertIsNone(result["meeting"]["requested_locale"])
        self.assertIn("Bonjour.", Path(result["meeting"]["readable_transcript_file"]).read_text())
        self.assertTrue(Path(result["meeting"]["transcript_file"]).is_file())
        self.credential_factory.return_value.__exit__.assert_called_once()

    def test_repeated_analysis_reuses_cache_and_restores_readable_transcript(self):
        self.recording()
        self.speech_transport()
        first = transcribe_meeting_file()
        Path(first["meeting"]["readable_transcript_file"]).unlink()
        second = transcribe_meeting_file("meeting.mp4")
        self.assertEqual(first["evidence"], second["evidence"])
        self.assertEqual(
            first["meeting"]["transcript_sha256"], second["meeting"]["transcript_sha256"],
        )
        self.assertEqual(first["meeting"]["evidence_source"], "azure-speech")
        self.assertEqual(second["meeting"]["evidence_source"], "cached-transcript")
        self.assertTrue(Path(second["meeting"]["readable_transcript_file"]).is_file())
        self.assertEqual(len(self.requests), 1)

    def test_concurrent_analysis_uses_one_speech_request(self):
        self.recording()
        self.speech_transport()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: transcribe_meeting_file(), range(2)))
        self.assertEqual(results[0]["evidence"], results[1]["evidence"])
        self.assertEqual(
            results[0]["meeting"]["transcript_sha256"],
            results[1]["meeting"]["transcript_sha256"],
        )
        self.assertEqual(len(self.requests), 1)

    def test_changed_recording_or_settings_invalidates_cache_and_report(self):
        path = self.recording()
        self.speech_transport()
        first = transcribe_meeting_file()
        result = save_meeting_report(
            "# Summary\n\nPilot approved. [U1 00:00:00]",
            first["meeting"]["transcript_file"], first["meeting"]["transcript_sha256"],
        )
        path.write_bytes(b"different")
        second = transcribe_meeting_file()
        self.assertNotEqual(first["meeting"]["transcript_sha256"], second["meeting"]["transcript_sha256"])
        self.assertFalse(Path(result["summary_file"]).exists())
        third = transcribe_meeting_file(locale="fr-FR", max_speakers=5)
        self.assertEqual(third["meeting"]["transcription_mode"], "fixed-locale")
        self.assertEqual(third["meeting"]["requested_locale"], "fr-FR")
        self.assertIn(b'"locales": ["fr-FR"]', self.requests[-1].content)
        self.assertEqual(len(self.requests), 3)

    def test_explicit_retranscription_bypasses_cache(self):
        self.recording()
        self.speech_transport()
        transcribe_meeting_file()
        transcribe_meeting_file(force_retranscribe=True)
        self.assertEqual(len(self.requests), 2)

    def test_failed_retranscription_preserves_previous_artifacts(self):
        recording = self.recording()
        self.speech_transport()
        meeting = transcribe_meeting_file()["meeting"]
        report = save_meeting_report(
            "Pilot approved. [U1]", meeting["transcript_file"], meeting["transcript_sha256"],
        )
        paths = [
            Path(meeting["transcript_file"]),
            Path(meeting["readable_transcript_file"]),
            Path(report["summary_file"]),
        ]
        before = {path: path.read_bytes() for path in paths}
        recording.write_bytes(b"changed recording")
        original_replace = Path.replace

        def replace(path, target):
            if path.suffix == ".new" and target == paths[1]:
                raise OSError("readable transcript publication failed")
            return original_replace(path, target)

        with patch.object(Path, "replace", replace), self.assertRaises(OSError):
            transcribe_meeting_file()
        self.assertEqual({path: path.read_bytes() for path in paths}, before)
        self.assertEqual(list(self.root.glob(".meeting-artifacts-*")), [])

    def test_corrupt_cache_is_not_silently_replaced(self):
        self.recording()
        self.recording("meeting.mp4.transcript.json", b"{}")
        self.speech_transport()
        with self.assertRaisesRegex(ValueError, "utterances"):
            transcribe_meeting_file()
        self.assertEqual(len(self.requests), 0)
        transcribe_meeting_file(force_retranscribe=True)
        self.assertEqual(len(self.requests), 1)

    def test_speech_failure_has_status_request_id_and_no_payload_or_artifacts(self):
        self.recording()
        self.speech_transport(status=403, payload={"error": "private service diagnostic"})
        with self.assertRaisesRegex(MeetingProcessingError, "HTTP 403.*test-request-123") as error:
            transcribe_meeting_file()
        self.assertNotIn("private", str(error.exception))
        self.assertEqual(list(self.root.glob("*.transcript.*")), [])

    def test_invalid_speech_response_does_not_persist_transcript(self):
        self.recording()
        self.speech_transport(payload={"phrases": []})
        with self.assertRaisesRegex(MeetingProcessingError, "invalid diarized"):
            transcribe_meeting_file()
        self.assertEqual(list(self.root.glob("*.transcript.*")), [])

    def test_missing_endpoint_fails_before_authentication(self):
        self.recording()
        self.speech_transport()
        with patch.dict(os.environ, {"AZURE_SPEECH_ENDPOINT": ""}):
            with self.assertRaisesRegex(MeetingProcessingError, "AZURE_SPEECH_ENDPOINT"):
                transcribe_meeting_file()
        self.credential_factory.assert_not_called()

    def test_recording_changed_in_flight_does_not_persist(self):
        self.recording()
        self.speech_transport()
        with patch("recording_tools._file_digest", side_effect=["before", "after"]):
            with self.assertRaisesRegex(MeetingProcessingError, "changed during"):
                transcribe_meeting_file()
        self.assertEqual(list(self.root.glob("*.transcript.*")), [])


class TranscriptArtifactTests(SessionTestCase):
    def test_inspects_uploaded_json_without_model_reserialization(self):
        path = self.root / "uploaded.json"
        path.write_text(TRANSCRIPT, encoding="utf-8")
        result = inspect_transcript(file_path="uploaded.json")
        self.assertEqual(result["meeting"]["evidence_source"], "user-supplied")
        self.assertTrue(Path(result["meeting"]["transcript_file"]).is_file())
        self.assertTrue(Path(result["meeting"]["readable_transcript_file"]).is_file())

    def test_pasted_json_persists_validated_artifacts(self):
        result = inspect_transcript(transcript_json=TRANSCRIPT)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(Path(result["meeting"]["transcript_file"]).is_file())

    def test_uploaded_provenance_cannot_claim_a_speech_request(self):
        supplied = json.loads(TRANSCRIPT)
        supplied["evidence_source"] = "azure-speech"
        transcript_json = json.dumps(supplied)
        (self.root / "uploaded.json").write_text(transcript_json, encoding="utf-8")
        for arguments in (
            {"transcript_json": transcript_json},
            {"file_path": "uploaded.json"},
        ):
            with self.subTest(arguments=arguments):
                result = inspect_transcript(**arguments)
                self.assertEqual(result["meeting"]["evidence_source"], "user-supplied")

    def test_requires_exactly_one_transcript_input(self):
        for kwargs in ({}, {"transcript_json": TRANSCRIPT, "file_path": "file.json"}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "either"):
                inspect_transcript(**kwargs)

    def test_saves_downloadable_summary_bound_to_validated_evidence(self):
        evidence = self.transcript()
        report = "# Summary\n\nPilot approved. [U1 00:00:00]"
        result = save_meeting_report(
            report, evidence["meeting"]["transcript_file"],
            evidence["meeting"]["transcript_sha256"],
        )
        saved = Path(result["summary_file"]).read_text(encoding="utf-8")
        self.assertEqual(result["report_markdown"], report)
        self.assertTrue(saved.startswith(report + "\n"))
        self.assertIn(evidence["meeting"]["transcript_sha256"], saved)
        reread = inspect_transcript(file_path=evidence["meeting"]["transcript_file"])
        self.assertEqual(reread["meeting"]["summary_file"], result["summary_file"])

    def test_rejects_missing_unknown_malformed_or_wrong_timestamp_citations(self):
        evidence = self.transcript()["meeting"]
        for report in (
            "No citations", "Wrong [U999 99:99:99]", "Wrong [U1 00:00:01]",
            "Wrong [U1 0:0:0]", "Mixed [U1 00:00:00] [U2 00:00:00]",
        ):
            with self.subTest(report=report), self.assertRaisesRegex(ValueError, "must cite"):
                save_meeting_report(report, evidence["transcript_file"], evidence["transcript_sha256"])
        self.assertEqual(list(self.root.glob("*.summary.md")), [])

    def test_rejects_invalid_transcript_or_stale_digest(self):
        evidence = self.transcript()["meeting"]
        with self.assertRaisesRegex(ValueError, "transcript changed"):
            save_meeting_report("[U1 00:00:00]", evidence["transcript_file"], "stale")
        Path(evidence["transcript_file"]).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "utterances"):
            save_meeting_report("[U1 00:00:00]", evidence["transcript_file"], evidence["transcript_sha256"])

    def test_unbound_old_summary_is_not_advertised(self):
        evidence = self.transcript()["meeting"]
        (self.root / "meeting.mp4.summary.md").write_text("Old report", encoding="utf-8")
        result = inspect_transcript(file_path=evidence["transcript_file"])
        self.assertNotIn("summary_file", result["meeting"])

    def test_publish_failure_preserves_existing_file_and_cleans_staging(self):
        path = self.root / "report.summary.md"
        path.write_text("Existing report", encoding="utf-8")
        with patch.object(Path, "replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                _publish_artifacts({path: "Replacement"})
        self.assertEqual(path.read_text(), "Existing report")
        self.assertEqual(list(self.root.glob(".meeting-artifacts-*")), [])

    def test_failed_first_ingestion_removes_partial_artifacts(self):
        original_replace = Path.replace

        def replace(path, target):
            if target.suffix == ".md":
                raise OSError("readable transcript publication failed")
            return original_replace(path, target)

        with patch.object(Path, "replace", replace), self.assertRaises(OSError):
            inspect_transcript(transcript_json=TRANSCRIPT)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_failed_report_removal_rolls_back_transcript_changes(self):
        paths = [self.root / name for name in ("transcript.json", "transcript.md", "summary.md")]
        for path in paths:
            path.write_text("Original", encoding="utf-8")
        original_unlink = Path.unlink

        def unlink(path, *args, **kwargs):
            if path == paths[2]:
                raise OSError("summary removal failed")
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", unlink), self.assertRaises(OSError):
            _publish_artifacts({paths[0]: "New JSON", paths[1]: "New Markdown", paths[2]: None})
        self.assertTrue(all(path.read_text() == "Original" for path in paths))

    def test_failed_rollback_retains_recovery_copies(self):
        first = self.root / "transcript.json"
        second = self.root / "transcript.md"
        first.write_text("Original JSON", encoding="utf-8")
        second.write_text("Original Markdown", encoding="utf-8")
        original_replace = Path.replace

        def replace(path, target):
            if target == second or path.suffix == ".old":
                raise OSError("storage unavailable")
            return original_replace(path, target)

        with patch.object(Path, "replace", replace), self.assertRaisesRegex(
            MeetingProcessingError, "Recovery copies remain",
        ):
            _publish_artifacts({first: "New JSON", second: "New Markdown"})
        staging = list(self.root.glob(".meeting-artifacts-*"))
        self.assertEqual(len(staging), 1)
        self.assertEqual(
            {path.read_text() for path in staging[0].glob("*.old")},
            {"Original JSON", "Original Markdown"},
        )


if __name__ == "__main__":
    unittest.main()
