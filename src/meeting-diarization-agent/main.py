import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from copy import copy
from pathlib import Path
from typing import Annotated, Any

import httpx
from agent_framework import (
    Agent,
    AgentContext,
    AgentResponse,
    AgentResponseUpdate,
    Content,
    Message,
    ResponseStream,
    tool,
)
from agent_framework.foundry import FoundryChatClient
from agent_framework_foundry_hosting import ResponsesHostServer
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pydantic import Field

from prompts import MEETING_INSTRUCTIONS
from recording_tools import (
    MeetingProcessingError,
    inspect_transcript,
    save_meeting_report,
    transcribe_meeting_file,
)

logger = logging.getLogger(__name__)


def _run_operation(operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return operation()
    except MeetingProcessingError as exc:
        code, message = exc.code, str(exc)
    except ValueError as exc:
        code, message = "invalid_input", str(exc)
    except ClientAuthenticationError:
        code, message = (
            "authentication_error",
            "Azure authentication failed. Check the configured identity and Speech RBAC.",
        )
    except httpx.RequestError:
        code, message = (
            "speech_connection_error",
            "The Speech request failed or timed out. No new evidence was returned. "
            "Check connectivity and service availability before retrying.",
        )
    except OSError:
        code, message = (
            "file_error",
            "A session file could not be read or saved. Check its availability, "
            "permissions and available storage. No download is confirmed.",
        )
    except Exception:
        logger.exception("Unexpected meeting tool failure")
        raise
    logger.warning("Meeting tool failed (%s): %s", code, message)
    return {"status": "error", "error": {"code": code, "message": message}}


@tool(approval_mode="never_require")
def inspect_diarized_transcript(
    transcript_json: Annotated[
        str,
        Field(
            description=(
                "A JSON object containing an utterances array. Each utterance must "
                "have speaker_id, start_ms, end_ms, and text. Optional fields include "
                "speaker_name, location, and overlaps_previous."
            )
        ),
    ] = "",
    file_path: Annotated[
        str,
        Field(
            description=(
                "Session-relative path of an uploaded or previously persisted "
                "transcript JSON. Prefer this to copying file contents into a tool call. "
                "Provide either file_path or transcript_json, not both."
            )
        ),
    ] = "",
) -> dict[str, Any]:
    """Ingest a diarized transcript for a new evidence-linked meeting analysis."""
    return _run_operation(lambda: inspect_transcript(transcript_json, file_path))


@tool(approval_mode="never_require")
def analyze_meeting_recording(
    file_path: Annotated[
        str,
        Field(
            description=(
                "Path or filename of an uploaded meeting recording. Use an empty "
                "string to automatically select the only uploaded recording."
            )
        ),
    ] = "",
    locale: Annotated[
        str,
        Field(
            description=(
                "Optional fixed BCP-47 locale such as fr-FR only when fixed-language "
                "transcription is explicitly requested. Use an empty string for "
                "multilingual meetings or when the language is unknown."
            )
        ),
    ] = "",
    max_speakers: Annotated[
        int,
        Field(
            description="Maximum expected number of speakers, from 2 through 35.",
            ge=2,
            le=35,
        ),
    ] = 10,
    force_retranscribe: Annotated[
        bool,
        Field(
            description=(
                "Reuse validated cached evidence by default. Set true only when "
                "the user explicitly requests retranscription."
            )
        ),
    ] = False,
) -> dict[str, Any]:
    """Transcribe an uploaded recording, or reuse matching validated evidence."""
    return _run_operation(
        lambda: transcribe_meeting_file(file_path, locale, max_speakers, force_retranscribe)
    )


@tool(approval_mode="never_require")
def get_meeting_files(
    file_path: Annotated[
        str,
        Field(
            description=(
                "The transcript_file path returned by a previous ingestion. "
                "Use for transcript/download follow-ups, not a new analysis."
            )
        ),
    ],
) -> dict[str, Any]:
    """Confirm available transcript and summary downloads without replacing the report."""
    return _run_operation(lambda: inspect_transcript(file_path=file_path))


def _replace_answer(response: AgentResponse, text: str) -> AgentResponse:
    messages = []
    for message in response.messages:
        if message.role != "assistant":
            messages.append(message)
            continue
        remaining = [content for content in message.contents if content.type != "text"]
        if remaining:
            preserved = copy(message)
            preserved.contents = remaining
            messages.append(preserved)
    messages.append(Message(role="assistant", contents=[text]))
    response.messages = messages
    return response


def _select_meeting(response: AgentResponse) -> tuple[dict[str, Any], bool]:
    calls: dict[str, str] = {}
    selected_meeting = None
    analysis_requested = False
    for message in response.messages:
        for content in message.contents:
            if content.type == "function_call":
                calls[content.call_id] = content.name
            elif content.type == "function_result":
                name = calls.get(content.call_id)
                if name not in {
                    "analyze_meeting_recording", "inspect_diarized_transcript",
                    "get_meeting_files",
                }:
                    continue
                if content.exception:
                    logger.error("Meeting tool execution failed: %s", name)
                    raise MeetingProcessingError(
                        "tool_execution_error", "Check the agent logs; "
                        "no new summary download is confirmed.",
                    )
                payload = json.loads(content.result)
                if payload["status"] == "error":
                    raise MeetingProcessingError(
                        payload["error"]["code"], payload["error"]["message"],
                    )
                meeting = payload["meeting"]
                if selected_meeting is not None and any(
                    meeting[key] != selected_meeting[key]
                    for key in ("transcript_file", "transcript_sha256")
                ):
                    raise MeetingProcessingError(
                        "multiple_transcripts",
                        "The tools returned different transcripts or versions. "
                        "Process one meeting per request. No new summary was saved.",
                    )
                selected_meeting = meeting
                analysis_requested |= name != "get_meeting_files"

    if selected_meeting is None:
        raise MeetingProcessingError(
            "missing_evidence",
            "No validated meeting evidence was returned. Upload a recording "
            "or transcript, then request an analysis or its saved downloads.",
        )
    return selected_meeting, analysis_requested


def _format_download_response(meeting: dict[str, Any]) -> str:
    summary = (
        f"\n\nExisting summary: `{Path(meeting['summary_file']).name}`."
        if meeting.get("summary_file") else "\n\nNo saved summary is confirmed."
    )
    return (
        f"Transcript confirmed: `{Path(meeting['readable_transcript_file']).name}`."
        f"{summary}\n\nDownload these files from the Playground Files tab."
    )


def _finalize_meeting_response(response: AgentResponse) -> AgentResponse:
    try:
        meeting, analysis_requested = _select_meeting(response)
    except MeetingProcessingError as exc:
        logger.warning("Meeting response withheld (%s): %s", exc.code, exc)
        return _replace_answer(
            response, f"Meeting processing failed ({exc.code}): {exc}",
        )
    if response.continuation_token or response.finish_reason in {"length", "content_filter"}:
        logger.warning("Incomplete meeting response; withholding the report")
        return _replace_answer(
            response, "The meeting response was incomplete. No new summary download "
            "is confirmed. Retry the analysis using the saved transcript.",
        )
    if not analysis_requested:
        return _replace_answer(response, _format_download_response(meeting))

    draft = next(
        (message.text for message in reversed(response.messages)
         if message.role == "assistant" and message.text.strip()),
        "",
    )
    saved = _run_operation(lambda: save_meeting_report(
        draft, meeting["transcript_file"], meeting["transcript_sha256"],
    ))
    if saved["status"] == "error":
        return _replace_answer(
            response, f"Report saving failed ({saved['error']['code']}): "
            f"{saved['error']['message']}",
        )
    readable_name = Path(meeting["readable_transcript_file"]).name
    return _replace_answer(
        response, f"{saved['report_markdown']}\n\n"
        f"**Downloads:** `{readable_name}` and `{Path(saved['summary_file']).name}`. "
        "Open the Playground Files tab to download them.",
    )


async def persist_response_middleware(
    context: AgentContext, call_next: Callable[[], Awaitable[None]],
) -> None:
    await call_next()
    if isinstance(context.result, ResponseStream):
        source = context.result

        async def release_verified_response() -> AsyncIterator[AgentResponseUpdate]:
            # Do not release model claims until their cited report is actually saved.
            response = await source.get_final_response()
            verified = await asyncio.to_thread(_finalize_meeting_response, response)
            for message in verified.messages:
                yield AgentResponseUpdate(
                    contents=message.contents, role=message.role,
                    message_id=message.message_id, author_name=message.author_name,
                    response_id=verified.response_id, agent_id=verified.agent_id,
                    created_at=verified.created_at,
                )
            if verified.usage_details:
                yield AgentResponseUpdate(
                    contents=[Content.from_usage(verified.usage_details)],
                    response_id=verified.response_id,
                )
            yield AgentResponseUpdate(
                response_id=verified.response_id, finish_reason=verified.finish_reason,
                continuation_token=verified.continuation_token,
            )

        context.result = ResponseStream(
            release_verified_response(), finalizer=AgentResponse.from_updates,
        )
    elif isinstance(context.result, AgentResponse):
        context.result = await asyncio.to_thread(_finalize_meeting_response, context.result)


def create_agent(credential: DefaultAzureCredential) -> Agent:
    required = ("FOUNDRY_PROJECT_ENDPOINT", "AZURE_AI_MODEL_DEPLOYMENT_NAME")
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")
    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        credential=credential,
    )

    return Agent(
        client=client,
        instructions=MEETING_INSTRUCTIONS,
        tools=[
            analyze_meeting_recording,
            inspect_diarized_transcript,
            get_meeting_files,
        ],
        middleware=[persist_response_middleware],
        default_options={"store": False, "tool_choice": "required"},
    )


def main():
    if not os.environ.get("FOUNDRY_HOSTING_ENVIRONMENT"):
        load_dotenv(Path(__file__).with_name(".env"))
    with DefaultAzureCredential() as credential:
        server = ResponsesHostServer(create_agent(credential))
        server.run()


if __name__ == "__main__":
    main()
