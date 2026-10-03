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

    def test_derives_overlap_for_unsorted_and_nested_intervals(self):
        payload = {"utterances": [
            {"speaker_id": "P1", "start_ms": 10000, "end_ms": 11000, "text": "Later."},
            {"speaker_id": "P2", "start_ms": 0, "end_ms": 9000, "text": "Long."},
            {"speaker_id": "P3", "start_ms": 500, "end_ms": 1000, "text": "Nested."},
            {"speaker_id": "P4", "start_ms": 2000, "end_ms": 3000, "text": "Also nested."},
        ]}
        result = json.loads(build_transcript_evidence(json.dumps(payload)))
        self.assertEqual(result["meeting"]["overlap_utterances"], ["U3", "U4"])
        self.assertTrue(result["evidence"][0].startswith("[U2 00:00:00]"))
        self.assertTrue(result["evidence"][-1].startswith("[U1 00:00:10]"))

    def test_flags_conflicting_supplied_overlap_metadata(self):
        payload = {"utterances": [
            {
                "speaker_id": "P1", "start_ms": 0, "end_ms": 1000,
                "text": "First.", "overlaps_previous": True,
            },
            {
                "speaker_id": "P2", "start_ms": 500, "end_ms": 1500,
                "text": "Overlapping.", "overlaps_previous": False,
            },
        ]}
        result = json.loads(build_transcript_evidence(json.dumps(payload)))
        self.assertEqual(result["meeting"]["overlap_utterances"], ["U2"])
        self.assertEqual(result["meeting"]["overlap_flag_conflicts"], ["U1", "U2"])

    def test_speech_order_does_not_create_false_overlap_or_change_ids_on_reload(self):
        payload = {"phrases": [
            {"speaker": 1, "offsetMilliseconds": 10000, "durationMilliseconds": 1000, "text": "Later."},
            {"speaker": 2, "offsetMilliseconds": 0, "durationMilliseconds": 1000, "text": "Earlier."},
        ]}
        converted = convert_speech_response(payload, "meeting.mp4")
        self.assertFalse(any(item["overlaps_previous"] for item in converted["utterances"]))
        first = build_transcript_evidence(json.dumps(converted))
        second = build_transcript_evidence(json.dumps(json.loads(json.dumps(converted))))
        self.assertEqual(first, second)
        self.assertEqual(json.loads(first)["meeting"]["overlap_utterances"], [])

    def test_rejects_non_object_response_and_invalid_speaker_ids(self):
        for payload in (None, [], "text"):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "JSON object"):
                convert_speech_response(payload, "meeting.mp4")
        for speaker in (None, True, -1, 1.5, "1", {}):
            payload = {"phrases": [{
                "speaker": speaker, "offsetMilliseconds": 0,
                "durationMilliseconds": 1000, "text": "Test.",
            }]}
            with self.subTest(speaker=speaker), self.assertRaisesRegex(ValueError, "speaker"):
                convert_speech_response(payload, "meeting.mp4")

    def test_rejects_invalid_confidence_and_locale(self):
        base = {
            "speaker": 0, "offsetMilliseconds": 0,
            "durationMilliseconds": 1000, "text": "Test.",
        }
        for confidence in (True, -0.1, 1.1, float("nan"), float("inf"), "0.5"):
            with self.subTest(confidence=confidence), self.assertRaisesRegex(ValueError, "confidence"):
                convert_speech_response({"phrases": [{**base, "confidence": confidence}]}, "meeting.mp4")
        for locale in (True, "", 123):
            with self.subTest(locale=locale), self.assertRaisesRegex(ValueError, "locale"):
                convert_speech_response({"phrases": [{**base, "locale": locale}]}, "meeting.mp4")

    def test_rejects_nonfinite_boolean_and_negative_timestamps(self):
        for value in (True, -1, float("nan"), float("inf")):
            payload = {"utterances": [{
                "speaker_id": "P1", "start_ms": value, "end_ms": 1000, "text": "Test.",
            }]}
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_transcript_evidence(json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
