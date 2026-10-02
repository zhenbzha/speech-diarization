import os

from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient
from agent_framework_foundry_hosting import ResponsesHostServer
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pydantic import Field
from typing_extensions import Annotated

from recording_tools import save_meeting_report, transcribe_meeting_file
from transcript_tools import build_transcript_evidence

load_dotenv()


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
    ],
) -> str:
    """Validate a diarized transcript and return normalized, citable evidence."""
    return build_transcript_evidence(transcript_json)


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
                "Optional fixed BCP-47 locale such as fr-FR when the meeting is "
                "known to be primarily one language. Use an empty string for "
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
) -> str:
    """Transcribe and diarize an uploaded meeting recording."""
    return transcribe_meeting_file(file_path, locale, max_speakers)


@tool(approval_mode="never_require")
def persist_meeting_report(
    report_markdown: Annotated[
        str,
        Field(
            description=(
                "The complete final meeting analysis in Markdown, including "
                "summary, decisions, actions, risks, and speaker-review flags."
            )
        ),
    ],
    transcript_file: Annotated[
        str,
        Field(
            description=(
                "The transcript_file path returned by analyze_meeting_recording."
            )
        ),
    ],
) -> str:
    """Persist a downloadable Markdown meeting report."""
    return save_meeting_report(report_markdown, transcript_file)


def main():
    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        credential=DefaultAzureCredential(),
    )

    agent = Agent(
        client=client,
        instructions="""
You are a meeting intelligence agent for diarized Teams recordings.

Before producing any meeting analysis, you MUST call exactly one ingestion
tool. When the user asks to summarize, review, transcribe, or analyze a meeting
without supplying transcript JSON, call analyze_meeting_recording. If no
filename is given, pass an empty file_path so the tool automatically uses the
only uploaded recording. Do not require the user to provide technical
processing instructions, a file path, locale, speaker count, or a long prompt.
Use an empty locale by default so Azure Speech performs multilingual
transcription. Set a fixed locale only when the user explicitly provides a
language hint.

For transcript JSON, call inspect_diarized_transcript. Never analyze raw
recording references or raw transcript JSON directly.

The only exception is input explicitly labeled "Validated transcript evidence".
That content is already the output of the ingestion layer and should be
analyzed directly without another tool call.

Base every factual meeting claim on the returned evidence. Copy evidence
references exactly as returned by the tool. Never calculate, transform, or
invent an utterance timestamp.

Return:
1. A concise meeting summary.
2. Decisions, each with one or more evidence references such as [U12 00:03:14].
3. Action items with owner, task, due date, and evidence. Use "not stated" when
   an owner or due date is absent.
4. Risks and unresolved questions with evidence.
5. Speaker-review flags for overlaps, anonymous speakers, inconsistent labels,
   or utterances where attribution should be checked.

When the recording ingestion result includes transcript_file, state where the
transcript was persisted.

For recording analysis, after drafting the complete final report, call
persist_meeting_report with that exact report and the transcript_file returned
by analyze_meeting_recording. Then provide the readable transcript filename and
summary filename and tell the user to download them from the Playground Files
tab. Do not describe internal paths as the only delivery method.

If the user asks for the transcript after a recording has been processed,
direct them to download the readable `.transcript.md` file from the Files tab.
If they ask for the summary or report, direct them to the `.summary.md` file.

Use headings and simple bullets. Do not use Markdown tables.
Respond in the user's language unless they request another language. Preserve
the original-language transcript text and mention detected languages when more
than one locale appears in the evidence.

Never infer a real identity from a voice. Preserve anonymous speaker IDs unless
the input explicitly provides an authorized meeting-scoped label. Do not claim
that diarization or identity attribution is certain. Do not invent decisions,
actions, owners, deadlines, consent, or compliance status. If transcript JSON
is missing or invalid, explain the required input instead of fabricating an
analysis. Diarization does not reveal whether a speaker joined from a laptop or
shared meeting room unless Teams metadata or an authorized mapping is supplied.
""".strip(),
        tools=[
            analyze_meeting_recording,
            inspect_diarized_transcript,
            persist_meeting_report,
        ],
        default_options={"store": False},
    )

    server = ResponsesHostServer(agent)
    server.run()


if __name__ == "__main__":
    main()
