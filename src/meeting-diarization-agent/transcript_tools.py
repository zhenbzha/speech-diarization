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
    if not math.isfinite(value) or not float(value).is_integer():
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

    overlaps_previous = value.get("overlaps_previous", False)
    if not isinstance(overlaps_previous, bool):
        raise ValueError(
            f"utterances[{index}].overlaps_previous must be a boolean when provided"
        )

    return {
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


def build_transcript_evidence(transcript_json: str) -> str:
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
        overlap = ", overlaps previous" if item["overlaps_previous"] else ""
        evidence_lines.append(
            f"[{item['utterance_id']} {item['timestamp']}] "
            f"{display_name} "
            f"({item['speaker_id']}{location}{locale}{overlap}): "
            f"{item['text']}"
        )

    result = {
        "meeting": {
            "title": payload.get("title"),
            "synthetic": payload.get("synthetic"),
            "duration_ms": max(item["end_ms"] for item in normalized),
            "utterance_count": len(normalized),
            "speaker_count": len(speaker_counts),
            "speaker_utterance_counts": dict(sorted(speaker_counts.items())),
            "anonymous_speakers": anonymous_speakers,
            "overlap_utterances": overlaps,
            "detected_locale_counts": dict(sorted(locale_counts.items())),
            "transcription_mode": payload.get("transcription_mode"),
            "requested_locale": payload.get("requested_locale"),
        },
        "evidence": evidence_lines,
    }
    return json.dumps(result, ensure_ascii=True, indent=2)


def convert_speech_response(
    payload: dict[str, Any], source_file: str
) -> dict[str, Any]:
    phrases = payload.get("phrases")
    if not isinstance(phrases, list) or not phrases:
        raise ValueError("Speech response must contain a non-empty phrases array")

    utterances = []
    previous_end_ms = 0
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
        if speaker is None or isinstance(speaker, bool):
            raise ValueError(
                f"phrases[{index}].speaker is missing; confirm diarization was enabled"
            )
        speaker_id = str(speaker).strip()
        if not speaker_id:
            raise ValueError(f"phrases[{index}].speaker must not be empty")
        if not speaker_id.lower().startswith("speaker"):
            speaker_id = f"Speaker {speaker_id}"

        end_ms = start_ms + duration_ms
        utterance = {
            "speaker_id": speaker_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text.strip(),
            "overlaps_previous": start_ms < previous_end_ms,
        }

        confidence = phrase.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            utterance["confidence"] = confidence

        locale = phrase.get("locale")
        if isinstance(locale, str) and locale.strip():
            utterance["locale"] = locale.strip()

        utterances.append(utterance)
        previous_end_ms = max(previous_end_ms, end_ms)

    return {
        "title": Path(source_file).stem,
        "synthetic": None,
        "source_file": source_file,
        "speech_duration_ms": payload.get("durationMilliseconds"),
        "utterances": utterances,
    }
