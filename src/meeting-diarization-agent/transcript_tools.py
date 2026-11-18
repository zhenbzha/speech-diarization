import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _format_timestamp(milliseconds: int) -> str:
    total_seconds = milliseconds // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _require_integer(value: Any, field: str, index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"utterances[{index}].{field} must be a whole number")
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        raise ValueError(f"utterances[{index}].{field} must be a whole number")
    integer_value = int(value)
    if integer_value < 0:
        raise ValueError(f"utterances[{index}].{field} must be non-negative")
    return integer_value


def _normalize_utterance(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"utterances[{index}] must be an object")

    speaker_id = value.get("speaker_id")
    if not isinstance(speaker_id, str) or not speaker_id.strip():
        raise ValueError(f"utterances[{index}].speaker_id must be a non-empty string")

    text = value.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"utterances[{index}].text must be a non-empty string")

    start_ms = _require_integer(value.get("start_ms"), "start_ms", index)
    end_ms = _require_integer(value.get("end_ms"), "end_ms", index)
    if end_ms <= start_ms:
        raise ValueError(f"utterances[{index}].end_ms must be greater than start_ms")

    speaker_name = value.get("speaker_name")
    if speaker_name is not None and (
        not isinstance(speaker_name, str) or not speaker_name.strip()
    ):
        raise ValueError(
            f"utterances[{index}].speaker_name must be a non-empty string when provided"
        )

    location = value.get("location")
    if location is not None and (
        not isinstance(location, str) or not location.strip()
    ):
        raise ValueError(
            f"utterances[{index}].location must be a non-empty string when provided"
        )

    locale = value.get("locale")
    if locale is not None and (
        not isinstance(locale, str) or not locale.strip()
    ):
        raise ValueError(
            f"utterances[{index}].locale must be a non-empty string when provided"
        )

    overlaps_previous = value.get("overlaps_previous")
    if "overlaps_previous" in value and not isinstance(overlaps_previous, bool):
        raise ValueError(
            f"utterances[{index}].overlaps_previous must be a boolean when provided"
        )

    confidence = value.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise ValueError(f"utterances[{index}].confidence must be between 0 and 1")

    result = {
        "utterance_id": f"U{index + 1}",
        "speaker_id": speaker_id.strip(),
        "speaker_name": speaker_name.strip() if speaker_name else None,
        "location": location.strip() if location else None,
        "locale": locale.strip() if locale else None,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "timestamp": _format_timestamp(start_ms),
        "overlaps_previous": overlaps_previous,
        "text": text.strip(),
    }
    if confidence is not None:
        result["confidence"] = confidence
    return result


def normalize_transcript(transcript_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(transcript_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"transcript_json is not valid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError("transcript_json must contain a JSON object")

    utterances = payload.get("utterances")
    if not isinstance(utterances, list) or not utterances:
        raise ValueError("transcript_json must contain a non-empty utterances array")

    normalized = [
        _normalize_utterance(utterance, index)
        for index, utterance in enumerate(utterances)
    ]
    normalized.sort(key=lambda item: (item["start_ms"], item["end_ms"]))

    previous_end_ms = 0
    for item in normalized:
        overlap = item["start_ms"] < previous_end_ms
        item["overlap_flag_conflict"] = (
            item["overlaps_previous"] is not None
            and item["overlaps_previous"] != overlap
        )
        item["overlaps_previous"] = overlap
        previous_end_ms = max(previous_end_ms, item["end_ms"])
    return {**payload, "utterances": normalized}


def build_transcript_evidence(transcript_json: str) -> dict[str, Any]:
    payload = normalize_transcript(transcript_json)
    normalized = payload["utterances"]
    speaker_counts = Counter(item["speaker_id"] for item in normalized)
    anonymous_speakers = sorted(
        {
            item["speaker_id"]
            for item in normalized
            if not item["speaker_name"]
        }
    )
    overlaps = [
        item["utterance_id"] for item in normalized if item["overlaps_previous"]
    ]
    locale_counts = Counter(
        item["locale"] for item in normalized if item["locale"]
    )

    evidence_lines = []
    for item in normalized:
        display_name = item["speaker_name"] or item["speaker_id"]
        location = f", {item['location']}" if item["location"] else ""
        locale = f", {item['locale']}" if item["locale"] else ""
        overlap = ", timestamp overlaps previous" if item["overlaps_previous"] else ""
        evidence_lines.append(
            f"[{item['utterance_id']} {item['timestamp']}] "
            f"{display_name} "
            f"({item['speaker_id']}{location}{locale}{overlap}): "
            f"{item['text']}"
        )

    return {
        "meeting": {
            "title": payload.get("title"),
            "synthetic": payload.get("synthetic"),
            "duration_ms": max(item["end_ms"] for item in normalized),
            "utterance_count": len(normalized),
            "speaker_count": len(speaker_counts),
            "speaker_utterance_counts": dict(sorted(speaker_counts.items())),
            "anonymous_speakers": anonymous_speakers,
            "overlap_utterances": overlaps,
            "overlap_flag_conflicts": [
                item["utterance_id"]
                for item in normalized
                if item["overlap_flag_conflict"]
            ],
            "detected_locale_counts": dict(sorted(locale_counts.items())),
            "transcription_mode": payload.get("transcription_mode"),
            "requested_locale": payload.get("requested_locale"),
        },
        "evidence": evidence_lines,
    }


def convert_speech_response(
    payload: dict[str, Any], source_file: str
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Speech response must be a JSON object")
    phrases = payload.get("phrases")
    if not isinstance(phrases, list) or not phrases:
        raise ValueError("Speech response must contain a non-empty phrases array")

    utterances = []
    for index, phrase in enumerate(phrases):
        if not isinstance(phrase, dict):
            raise ValueError(f"phrases[{index}] must be an object")

        start_ms = _require_integer(
            phrase.get("offsetMilliseconds"), "offsetMilliseconds", index
        )
        duration_ms = _require_integer(
            phrase.get("durationMilliseconds"), "durationMilliseconds", index
        )
        if duration_ms == 0:
            raise ValueError(
                f"phrases[{index}].durationMilliseconds must be greater than zero"
            )

        text = phrase.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"phrases[{index}].text must be a non-empty string")

        speaker = phrase.get("speaker")
        if (
            isinstance(speaker, bool)
            or not isinstance(speaker, int)
            or speaker < 0
        ):
            raise ValueError(
                f"phrases[{index}].speaker must be a non-negative integer; "
                "confirm diarization was enabled"
            )

        end_ms = start_ms + duration_ms
        utterance = {
            "speaker_id": f"Speaker {speaker}",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text.strip(),
        }

        for field in ("confidence", "locale"):
            if field in phrase:
                utterance[field] = phrase[field]

        utterances.append(utterance)

    result = {
        "title": Path(source_file).stem,
        "synthetic": None,
        "source_file": source_file,
        "speech_duration_ms": payload.get("durationMilliseconds"),
        "utterances": utterances,
    }
    # Validate before persistence, but retain source order so evidence IDs remain
    # stable when the persisted JSON is read again.
    normalized = normalize_transcript(json.dumps(result))
    for item in normalized["utterances"]:
        index = int(item["utterance_id"][1:]) - 1
        utterances[index]["overlaps_previous"] = item["overlaps_previous"]
    return result
