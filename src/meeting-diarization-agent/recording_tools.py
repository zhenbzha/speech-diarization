import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from threading import RLock
from urllib.parse import urlsplit

import httpx
from azure.identity import DefaultAzureCredential

from transcript_tools import build_transcript_evidence, convert_speech_response


ALLOWED_SUFFIXES = {".mp4", ".wav", ".mp3", ".m4a", ".aac", ".webm"}
MAX_FILE_BYTES = 500_000_000
SPEECH_API_VERSION = "2025-10-15"
TRANSCRIPT_SUFFIX = ".transcript.json"
CACHE_VERSION = 1
# Foundry gives each session its own sandbox. Serialize its file transactions
# across synchronous tools, which Agent Framework runs on worker threads.
_artifact_lock = RLock()


class MeetingProcessingError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _session_root() -> Path:
    root = Path(os.environ.get("MEETING_SESSION_DIR") or Path.home()).resolve()
    if not root.is_dir():
        raise ValueError("The configured meeting session directory does not exist.")
    return root


def _contained_path(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(_session_root()):
        raise ValueError("Meeting files must stay inside the hosted session directory.")
    return resolved


def _choose_file(paths: list[Path]) -> Path:
    contained = {_contained_path(path) for path in paths}
    matches = sorted(path for path in contained if path.is_file())
    if not matches:
        raise ValueError(
            "No matching uploaded meeting file was found. Upload it in the "
            "session's Files tab, then provide its session-relative path."
        )
    if len(matches) > 1:
        names = ", ".join(str(path.relative_to(_session_root())) for path in matches)
        raise ValueError(
            "Multiple uploaded recordings or transcripts match. "
            f"Specify a session-relative path: {names}"
        )
    return matches[0]


def _resolve_session_path(file_path: str) -> Path:
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("A session file path is required.")
    file_path = file_path.strip()
    root = _session_root()
    candidate = Path(file_path)
    if ".." in candidate.parts:
        raise ValueError("Parent traversal is not allowed in a session directory.")

    # Accept earlier /data aliases under $HOME, never as host filesystem paths.
    if file_path.startswith(("/data/", "\\data\\")):
        paths = [root / file_path.lstrip("/\\")]
    elif candidate.is_absolute() or candidate.anchor:
        paths = [_contained_path(candidate)]
    else:
        paths = [root / candidate]
        if len(candidate.parts) == 1:
            paths.append(root / "data" / candidate)
    return _choose_file(paths)


def _validate_recording(path: Path) -> Path:
    path = _contained_path(path)
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(
            f"Unsupported meeting file type '{path.suffix}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_SUFFIXES))}"
        )
    size = path.stat().st_size
    if size == 0:
        raise ValueError("The uploaded recording is empty.")
    if size >= MAX_FILE_BYTES:
        raise ValueError("The recording must be smaller than 500 MB for transcription.")
    return path


def discover_uploaded_recording() -> Path:
    root = _session_root()
    paths = []
    for directory in (root, root / "data"):
        directory = _contained_path(directory)
        if directory.is_dir():
            paths.extend(
                path for path in directory.iterdir()
                if path.suffix.lower() in ALLOWED_SUFFIXES
            )
    return _validate_recording(_choose_file(paths))


def resolve_session_file(file_path: str) -> Path:
    if not file_path.strip():
        return discover_uploaded_recording()
    return _validate_recording(_resolve_session_path(file_path))


def build_transcription_definition(locale: str, max_speakers: int) -> dict[str, object]:
    if (
        isinstance(max_speakers, bool)
        or not isinstance(max_speakers, int)
        or not 2 <= max_speakers <= 35
    ):
        raise ValueError("max_speakers must be an integer between 2 and 35")
    if not isinstance(locale, str):
        raise ValueError("locale must be a string")
    locale = locale.strip()
    if locale and not re.fullmatch(r"[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})+", locale):
        raise ValueError("Use a BCP-47 locale such as fr-FR, or leave locale empty.")
    definition: dict[str, object] = {
        "diarization": {"enabled": True, "maxSpeakers": max_speakers},
        "profanityFilterMode": "Masked",
    }
    if locale:
        definition["locales"] = [locale]
    return definition


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    _contained_path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _report_path(transcript_path: Path) -> Path:
    return transcript_path.with_name(
        transcript_path.name.removesuffix(TRANSCRIPT_SUFFIX) + ".summary.md"
    )


def _persist_evidence(transcript_json: str, transcript_path: Path) -> str:
    evidence = json.loads(build_transcript_evidence(transcript_json))
    _contained_path(transcript_path)
    report_path = _report_path(transcript_path)
    if (
        transcript_path.exists()
        and transcript_path.read_text(encoding="utf-8") != transcript_json
    ):
        _contained_path(report_path)
        report_path.unlink(missing_ok=True)
    readable_path = transcript_path.with_suffix(".md")
    _atomic_write(
        readable_path,
        "# Meeting transcript\n\n" + "\n\n".join(evidence["evidence"]) + "\n",
    )
    _atomic_write(transcript_path, transcript_json)
    digest = _text_digest(transcript_json)
    evidence["status"] = "ok"
    evidence["meeting"].update(
        transcript_file=str(transcript_path),
        transcript_sha256=digest,
        readable_transcript_file=str(readable_path),
        evidence_source=json.loads(transcript_json).get("evidence_source", "user-supplied"),
    )
    if report_path.is_file():
        _contained_path(report_path)
        marker = f"<!-- transcript-sha256: {digest} -->"
        if report_path.read_text(encoding="utf-8").endswith(marker + "\n"):
            evidence["meeting"]["summary_file"] = str(report_path)
    return json.dumps(evidence, ensure_ascii=True, indent=2)


def inspect_transcript(transcript_json: str = "", file_path: str = "") -> str:
    with _artifact_lock:
        if bool(transcript_json.strip()) == bool(file_path.strip()):
            raise ValueError("Provide either transcript_json or a transcript file path.")
        if file_path.strip():
            source = _resolve_session_path(file_path)
            if source.suffix.lower() != ".json":
                raise ValueError("The uploaded transcript must be a JSON file.")
            transcript_json = source.read_text(encoding="utf-8")
            target = (
                source if source.name.endswith(TRANSCRIPT_SUFFIX)
                else source.with_suffix(TRANSCRIPT_SUFFIX)
            )
        else:
            target = _session_root() / (
                f"supplied-{_text_digest(transcript_json)[:16]}{TRANSCRIPT_SUFFIX}"
            )
        # Validation is structural, not proof that user-supplied statements are true.
        return _persist_evidence(transcript_json, target)


def _speech_endpoint() -> str:
    endpoint = os.environ.get("AZURE_SPEECH_ENDPOINT", "").strip().rstrip("/")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https" or not parsed.hostname
        or parsed.username or parsed.password or parsed.query or parsed.fragment
        or parsed.path
    ):
        raise MeetingProcessingError(
            "configuration_error",
            "Set AZURE_SPEECH_ENDPOINT to the HTTPS endpoint of the Speech resource.",
        )
    return endpoint


def transcribe_meeting_file(
    file_path: str = "",
    locale: str = "",
    max_speakers: int = 10,
    force_retranscribe: bool = False,
) -> str:
    definition = build_transcription_definition(locale, max_speakers)
    with _artifact_lock:
        path = resolve_session_file(file_path)
        fingerprint = {
            "version": CACHE_VERSION,
            "recording_sha256": _file_digest(path),
            "definition": definition,
            "api_version": SPEECH_API_VERSION,
        }
        transcript_path = path.with_name(path.name + TRANSCRIPT_SUFFIX)
        if transcript_path.is_file() and not force_retranscribe:
            _contained_path(transcript_path)
            cached_json = transcript_path.read_text(encoding="utf-8")
            build_transcript_evidence(cached_json)
            if json.loads(cached_json).get("transcription_request") == fingerprint:
                return _persist_evidence(cached_json, transcript_path)

        endpoint = _speech_endpoint()
        with DefaultAzureCredential() as credential, httpx.Client(
            timeout=600.0
        ) as client, path.open("rb") as stream:
            token = credential.get_token("https://cognitiveservices.azure.com/.default")
            response = client.post(
                f"{endpoint}/speechtotext/transcriptions:transcribe"
                f"?api-version={SPEECH_API_VERSION}",
                headers={"Authorization": f"Bearer {token.token}"},
                files={
                    "audio": (
                        path.name, stream,
                        "video/mp4" if path.suffix.lower() == ".mp4"
                        else "application/octet-stream",
                    )
                },
                data={"definition": json.dumps(definition)},
            )
        if not response.is_success:
            request_id = response.headers.get("apim-request-id", "not provided")
            request_id = re.sub(r"[^a-zA-Z0-9 ._-]", "", request_id)[:128]
            raise MeetingProcessingError(
                "speech_service_error",
                f"Azure Speech returned HTTP {response.status_code}; "
                f"request ID: {request_id}. Check Speech access, supported "
                "locale/format, service limits and availability before retrying.",
            )
        try:
            transcript = convert_speech_response(response.json(), str(path))
        except ValueError as exc:
            raise MeetingProcessingError(
                "invalid_speech_response",
                "Azure Speech returned invalid diarized transcript data. "
                "No new transcript or summary was saved.",
            ) from exc
        if _file_digest(path) != fingerprint["recording_sha256"]:
            raise MeetingProcessingError(
                "recording_changed",
                "The recording changed during transcription. Submit the current file again.",
            )
        transcript.update(
            evidence_source="azure-speech",
            transcription_mode="fixed-locale" if locale.strip() else "multilingual",
            requested_locale=locale.strip() or None,
            transcription_request=fingerprint,
        )
        return _persist_evidence(
            json.dumps(transcript, ensure_ascii=True, indent=2), transcript_path
        )


def save_meeting_report(
    report_markdown: str, transcript_file: str, transcript_sha256: str
) -> str:
    if not report_markdown.strip():
        raise ValueError("The meeting report must not be empty.")
    with _artifact_lock:
        transcript_path = _resolve_session_path(transcript_file)
        if not transcript_path.name.endswith(TRANSCRIPT_SUFFIX):
            raise ValueError("The report source must be a persisted transcript JSON file.")
        transcript_json = transcript_path.read_text(encoding="utf-8")
        evidence = json.loads(build_transcript_evidence(transcript_json))
        digest = _text_digest(transcript_json)
        if digest != transcript_sha256:
            raise ValueError("The transcript changed. Ingest it again before saving a report.")
        valid_references = {line.split("]", 1)[0] + "]" for line in evidence["evidence"]}
        canonical_references = {
            reference.split(" ", 1)[0] + "]": reference for reference in valid_references
        }
        references = re.findall(r"\[U[^\]\r\n]*\]", report_markdown)
        invalid_references = [
            ref for ref in references
            if ref not in valid_references and ref not in canonical_references
        ]
        if not references or invalid_references:
            detail = (
                "Invalid references: " + ", ".join(
                    repr(ref[:120]) for ref in sorted(set(invalid_references))[:8]
                )
                if invalid_references else "No [U#] references were found."
            )
            raise ValueError(
                "The report must cite existing [U#] evidence IDs or exact "
                "[U# hh:mm:ss] references. At least one reference is required; "
                f"all must exist. {detail}"
            )
        report_markdown = re.sub(
            r"\[U[^\]\r\n]*\]",
            lambda match: canonical_references.get(match.group(), match.group()),
            report_markdown,
        )
        report_path = _report_path(transcript_path)
        _atomic_write(
            report_path,
            report_markdown.strip() + f"\n\n<!-- transcript-sha256: {digest} -->\n",
        )
        return json.dumps(
            {
                "status": "ok",
                "summary_file": str(report_path),
                "transcript_sha256": digest,
                "report_markdown": report_markdown.strip(),
                "download_instruction": (
                    "Open the Files tab in the Foundry Playground and download "
                    f"{report_path.name}."
                ),
            },
            ensure_ascii=True,
        )
