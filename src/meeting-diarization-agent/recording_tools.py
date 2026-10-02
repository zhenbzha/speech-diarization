import json
import os
from pathlib import Path

import httpx
from azure.identity import DefaultAzureCredential

from transcript_tools import build_transcript_evidence, convert_speech_response


ALLOWED_SUFFIXES = {".mp4", ".wav", ".mp3", ".m4a", ".aac", ".webm"}
MAX_FILE_BYTES = 500 * 1024 * 1024


def _session_roots() -> list[Path]:
    candidates = [
        Path.home(),
        Path.cwd(),
        Path("/data"),
        Path("/mnt/data"),
        Path("/home/oai/share"),
    ]
    roots = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def discover_uploaded_recording() -> Path:
    recordings = []
    for root in _session_roots():
        directories = [root, root / "data"]
        for directory in directories:
            if not directory.is_dir():
                continue
            for path in directory.iterdir():
                if (
                    path.is_file()
                    and path.suffix.lower() in ALLOWED_SUFFIXES
                    and path not in recordings
                ):
                    recordings.append(path.resolve())

    if not recordings:
        raise ValueError(
            "No uploaded meeting recording was found in the hosted session."
        )
    if len(recordings) > 1:
        names = ", ".join(path.name for path in recordings)
        raise ValueError(
            "Multiple uploaded recordings were found. Specify one filename: "
            f"{names}"
        )
    return recordings[0]


def _resolve_session_path(file_path: str) -> Path:
    candidate = Path(file_path).expanduser()
    roots = _session_roots()
    is_virtual_absolute = str(candidate).startswith(("/", "\\"))
    if candidate.is_absolute() or is_virtual_absolute:
        virtual_path = Path(str(candidate).lstrip("/\\"))
        paths = [candidate.resolve()]
        paths.extend((root / virtual_path).resolve() for root in roots)
    else:
        paths = [(root / candidate).resolve() for root in roots]
    path = next((item for item in paths if item.is_file()), paths[0])

    if not any(path == root or root in path.parents for root in roots):
        raise ValueError(
            "Meeting files must be uploaded into an allowed hosted session directory."
        )

    if not path.is_file():
        searched = ", ".join(str(item) for item in paths)
        raise ValueError(f"Meeting file does not exist. Searched: {searched}")
    return path


def resolve_session_file(file_path: str) -> Path:
    if not file_path.strip():
        return discover_uploaded_recording()

    path = _resolve_session_path(file_path)
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(
            f"Unsupported meeting file type '{path.suffix}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_SUFFIXES))}"
        )
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Meeting file exceeds the 500 MB transcription limit.")
    return path


def build_transcription_definition(
    locale: str, max_speakers: int
) -> dict[str, object]:
    definition: dict[str, object] = {
        "diarization": {
            "enabled": True,
            "maxSpeakers": max_speakers,
        }
    }
    if locale.strip():
        definition["locales"] = [locale.strip()]
    return definition


def transcribe_meeting_file(
    file_path: str = "",
    locale: str = "",
    max_speakers: int = 10,
) -> str:
    if not 2 <= max_speakers <= 35:
        raise ValueError("max_speakers must be between 2 and 35")

    path = resolve_session_file(file_path)
    endpoint = os.environ["AZURE_SPEECH_ENDPOINT"].rstrip("/")
    token = DefaultAzureCredential().get_token(
        "https://cognitiveservices.azure.com/.default"
    )
    definition = build_transcription_definition(locale, max_speakers)

    with path.open("rb") as stream:
        response = httpx.post(
            (
                f"{endpoint}/speechtotext/transcriptions:transcribe"
                "?api-version=2025-10-15"
            ),
            headers={"Authorization": f"Bearer {token.token}"},
            files={
                "audio": (
                    path.name,
                    stream,
                    "video/mp4"
                    if path.suffix.lower() == ".mp4"
                    else "application/octet-stream",
                )
            },
            data={"definition": json.dumps(definition)},
            timeout=600.0,
        )

    if response.is_error:
        request_id = response.headers.get("apim-request-id", "not provided")
        raise RuntimeError(
            "Azure Speech transcription failed "
            f"with HTTP {response.status_code}; request ID: {request_id}; "
            f"response: {response.text[:1000]}"
        )

    transcript = convert_speech_response(response.json(), str(path))
    transcript["transcription_mode"] = (
        "fixed-locale" if locale.strip() else "multilingual"
    )
    transcript["requested_locale"] = locale.strip() or None
    transcript_path = path.with_suffix(path.suffix + ".transcript.json")
    transcript_path.write_text(
        json.dumps(transcript, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    evidence = json.loads(build_transcript_evidence(json.dumps(transcript)))
    readable_transcript_path = path.with_suffix(
        path.suffix + ".transcript.md"
    )
    readable_transcript_path.write_text(
        "# Meeting transcript\n\n"
        + "\n\n".join(evidence["evidence"])
        + "\n",
        encoding="utf-8",
    )
    evidence["meeting"]["transcript_file"] = str(transcript_path)
    evidence["meeting"]["readable_transcript_file"] = str(
        readable_transcript_path
    )
    return json.dumps(evidence, ensure_ascii=True, indent=2)


def save_meeting_report(report_markdown: str, transcript_file: str) -> str:
    if not report_markdown.strip():
        raise ValueError("The meeting report must not be empty.")

    transcript_path = _resolve_session_path(transcript_file)
    if not transcript_path.name.endswith(".transcript.json"):
        raise ValueError("The report source must be a persisted transcript JSON file.")
    suffix = ".transcript.json"
    if transcript_path.name.endswith(suffix):
        base_name = transcript_path.name[: -len(suffix)]
    else:
        base_name = transcript_path.stem
    report_path = transcript_path.with_name(f"{base_name}.summary.md")
    report_path.write_text(report_markdown.strip() + "\n", encoding="utf-8")
    return json.dumps(
        {
            "summary_file": str(report_path),
            "download_instruction": (
                "Open the Files tab in the Foundry Playground and download "
                f"{report_path.name}."
            ),
        },
        ensure_ascii=True,
    )
