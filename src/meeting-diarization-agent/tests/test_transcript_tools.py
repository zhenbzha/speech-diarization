import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transcript_tools import build_transcript_evidence, convert_speech_response


class BuildTranscriptEvidenceTests(unittest.TestCase):
    def test_builds_citable_evidence_and_statistics(self):
        payload = {
            "title": "Pilot review",
            "synthetic": True,
            "utterances": [
                {
                    "speaker_id": "P1",
                    "speaker_name": "Alex",
                    "location": "Meeting room",
                    "start_ms": 1200,
                    "end_ms": 3500,
                    "text": "We will start with internal recordings.",
                },
                {
                    "speaker_id": "P2",
                    "start_ms": 3000,
                    "end_ms": 5200,
                    "overlaps_previous": True,
                    "text": "Security review remains required.",
                },
            ],
        }

        result = json.loads(build_transcript_evidence(json.dumps(payload)))

        self.assertEqual(result["meeting"]["speaker_count"], 2)
        self.assertEqual(result["meeting"]["detected_locale_counts"], {})
        self.assertEqual(result["meeting"]["anonymous_speakers"], ["P2"])
        self.assertEqual(result["meeting"]["overlap_utterances"], ["U2"])
        self.assertIn("[U1 00:00:01] Alex", result["evidence"][0])
        self.assertIn("overlaps previous", result["evidence"][1])

    def test_rejects_invalid_time_range(self):
        payload = {
            "utterances": [
                {
                    "speaker_id": "P1",
                    "start_ms": 1000,
                    "end_ms": 1000,
                    "text": "Invalid range.",
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "greater than start_ms"):
            build_transcript_evidence(json.dumps(payload))

    def test_rejects_missing_utterances(self):
        with self.assertRaisesRegex(ValueError, "non-empty utterances array"):
            build_transcript_evidence("{}")

    def test_rejects_non_boolean_overlap_flag(self):
        payload = {
            "utterances": [
                {
                    "speaker_id": "P1",
                    "start_ms": 0,
                    "end_ms": 1000,
                    "overlaps_previous": "false",
                    "text": "Invalid overlap flag.",
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            build_transcript_evidence(json.dumps(payload))

    def test_accepts_integral_json_float_timestamps(self):
        payload = {
            "utterances": [
                {
                    "speaker_id": "P1",
                    "start_ms": 0.0,
                    "end_ms": 1000.0,
                    "text": "Whole-number floats are valid millisecond values.",
                }
            ]
        }

        result = json.loads(build_transcript_evidence(json.dumps(payload)))

        self.assertEqual(result["meeting"]["duration_ms"], 1000)

    def test_rejects_fractional_timestamp(self):
        payload = {
            "utterances": [
                {
                    "speaker_id": "P1",
                    "start_ms": 0,
                    "end_ms": 1000.5,
                    "text": "Fractional milliseconds are outside the contract.",
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "must be a whole number"):
            build_transcript_evidence(json.dumps(payload))

    def test_converts_speech_diarization_response(self):
        payload = {
            "durationMilliseconds": 2500,
            "phrases": [
                {
                    "speaker": 1,
                    "offsetMilliseconds": 100,
                    "durationMilliseconds": 1000,
                    "text": "First speaker.",
                    "locale": "en-US",
                    "confidence": 0.9,
                },
                {
                    "speaker": 2,
                    "offsetMilliseconds": 900,
                    "durationMilliseconds": 1000,
                    "text": "Second speaker overlaps.",
                    "locale": "en-US",
                },
            ],
        }

        result = convert_speech_response(payload, "meeting.mp4")

        self.assertEqual(result["utterances"][0]["speaker_id"], "Speaker 1")
        self.assertEqual(result["utterances"][0]["locale"], "en-US")
        self.assertTrue(result["utterances"][1]["overlaps_previous"])

    def test_reports_detected_locale_counts(self):
        payload = {
            "utterances": [
                {
                    "speaker_id": "P1",
                    "start_ms": 0,
                    "end_ms": 1000,
                    "locale": "en-US",
                    "text": "Hello.",
                },
                {
                    "speaker_id": "P2",
                    "start_ms": 1200,
                    "end_ms": 2200,
                    "locale": "fr-FR",
                    "text": "Bonjour.",
                },
            ]
        }

        result = json.loads(build_transcript_evidence(json.dumps(payload)))

        self.assertEqual(
            result["meeting"]["detected_locale_counts"],
            {"en-US": 1, "fr-FR": 1},
        )
        self.assertIn("fr-FR", result["evidence"][1])


if __name__ == "__main__":
    unittest.main()
