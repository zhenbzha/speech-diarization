# Meeting Diarization Agent

Turn Teams meeting recordings into speaker-labeled transcripts and structured
meeting summaries. This Python application combines **Azure Speech** for
transcription and diarization with **GPT-4.1** for meeting analysis, hosted in
**Microsoft Foundry Agent Service**.

The agent produces summaries, decisions, action items, and open questions, with
timestamped references back to the transcript. It supports multilingual
transcription and reuses saved transcripts for follow-up analysis.

## Try it in Foundry

After deploying the agent:

1. Open `meeting-diarization-agent` in the Foundry portal and select **Playground**.
2. Upload a meeting recording through the session's **Files** tab.
3. Send **"Summarize the meeting."**
4. Download the transcript and summary from **Files**. Refresh the file list if
   the generated files are not visible yet.

For a ready-to-use example, upload
`assets\foundry-iq-hr-sales-meeting.mp4`. The approximately 12 MB synthetic
recording features five fictional participants discussing an HR assistant pilot.

Other example prompts:

- "Summarize the meeting and highlight decisions, action items, and open questions."
- "Summarize the meeting in French."
- "Can I download the transcript?"

Keep follow-up requests in the same session. If you upload several recordings,
include the filename in your prompt.

### Generated files

For an uploaded `meeting.mp4`, the agent saves:

- **`meeting.mp4.transcript.md`** -- readable transcript with speaker labels and
  timestamps.
- **`meeting.mp4.transcript.json`** -- normalized phrase-level transcript with
  timing, speaker IDs, and transcription metadata.
- **`meeting.mp4.summary.md`** -- meeting summary, decisions, action items, risks,
  and unresolved questions.

The final answer appears after the report has been validated and saved. Citation
or processing errors are reported in chat rather than publishing an invalid
report. Files remain in the hosted session, separate from the application code.

## How it works

```text
Meeting recording
    -> Azure Speech: transcription, speaker diarization, timestamps
    -> Python: transcript normalization and evidence references
    -> GPT-4.1: summary, decisions, actions, and open questions
    -> Python: citation validation and downloadable files
```

**Azure Speech** processes the recording through the Fast Transcription REST API
(`2025-10-15`). The application sends the file directly using `httpx` and
authenticates with Microsoft Entra ID through `DefaultAzureCredential`.

**Transcript processing** preserves Speech's text and timing, formats speaker
labels, and adds utterance references such as `U12`. Timestamps come from Speech,
not the language model.

**Meeting analysis** uses Microsoft Agent Framework and a Foundry model
deployment. The model cites utterance IDs; application middleware validates those
IDs, inserts the corresponding timestamps, and saves the report before returning
it. Repeated analysis reuses the transcript when the recording and transcription
settings are unchanged.

Multilingual transcription is the default. A request to change the summary's
language does not change the original-language transcript.

## Code structure

```text
speech-diarization\
  azure.yaml                         Foundry deployment configuration
  assets\
    foundry-iq-hr-sales-meeting.mp4    Synthetic demo recording
  src\
    meeting-diarization-agent\
      main.py                        Agent instructions, tools, and response handling
      recording_tools.py             Speech API calls, session files, caching, report saving
      transcript_tools.py            Transcript conversion, validation, and evidence
      pyproject.toml                 Python dependencies
      uv.lock                        Pinned dependency versions and hashes
      .env.example                   Configuration template
      tests\                         Unit tests
```

Start with **`main.py`** to understand the agent. The Speech API call is in
**`recording_tools.py`**, inside `transcribe_meeting_file()`.

## Local development

### Prerequisites

- Python 3.13 and [uv](https://docs.astral.sh/uv/).
- A Microsoft Foundry project with a `gpt-4.1` deployment, or another compatible
  model configured through `AZURE_AI_MODEL_DEPLOYMENT_NAME`.
- An Azure Speech resource in a region that supports Fast Transcription.
- An Azure identity with model-inference access and the **Cognitive Services User**
  role on the Speech resource. See [Speech access control](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/role-based-access-control).
- [Azure Developer CLI](https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd)
  and the agent extension required by `azure.yaml` for local invocation and deployment.

The commands below use PowerShell and start from the repository root.

### Install dependencies

```powershell
Set-Location .\src\meeting-diarization-agent
```

Install the versions pinned in `uv.lock` using your organization's approved
package source. On Microsoft-managed devices, use the protected-feed option
below instead of the standard command.

**Standard installation**

```powershell
uv sync --locked --python 3.13
```

<details>
<summary>Microsoft-managed devices: install through the protected feed</summary>

```powershell
uv venv --python 3.13
uv export --frozen --no-dev --no-emit-project --format requirements-txt |
    uv pip sync --python .\.venv\Scripts\python.exe --require-hashes `
        --default-index https://packagefeedproxy.microsoft.io/pypi/simple -
```

This installs the same locked versions with hash verification through the
approved feed, without changing the cloud build's dependency configuration.

</details>

### Configure and run

Copy the configuration template:

```powershell
Copy-Item .env.example .env
```

Set these values in `.env`:

- `FOUNDRY_PROJECT_ENDPOINT` -- your Foundry project endpoint.
- `AZURE_AI_MODEL_DEPLOYMENT_NAME` -- your model deployment name.
- `AZURE_SPEECH_ENDPOINT` -- your Speech resource endpoint.

Sign in with an Azure identity supported by `DefaultAzureCredential`, such as
your Azure CLI identity. No Speech API key is required.

Create a local session directory, add the sample recording, and start the agent:

```powershell
New-Item -ItemType Directory -Path .\artifacts -Force | Out-Null
Copy-Item ..\..\assets\foundry-iq-hr-sales-meeting.mp4 .\artifacts\
$env:MEETING_SESSION_DIR = (Resolve-Path .\artifacts).Path
uv run --no-sync python main.py
```

In another terminal, invoke the local agent from the repository root:

```powershell
azd ai agent invoke --local "Summarize the meeting."
```

Local transcripts and summaries are written to the `artifacts` directory.
The `--no-sync` option uses the prepared Python environment without restoring
dependencies again.

### Run tests

From `src\meeting-diarization-agent`:

```powershell
uv run --no-sync python -m unittest discover -s tests -v
```

The tests cover transcript conversion, file handling, caching, citations, and
report persistence. Azure calls are mocked; running the suite does not require
live service access.

## Deploy to Foundry

The root `azure.yaml` defines a Python 3.13 hosted agent using the Responses
protocol and remote dependency builds.

Configure an `azd` environment for your target Foundry project following the
[hosted-agent deployment guide](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/deploy-hosted-agent-code).
Set `AZURE_AI_MODEL_DEPLOYMENT_NAME` and `AZURE_SPEECH_ENDPOINT` in that
environment, and grant the hosted agent's identity access to the model and Speech.
Review the model and capacity settings in `azure.yaml` before provisioning resources.

Once the environment and resources are configured, deploy from the repository root:

```powershell
azd deploy
```

Leave `MEETING_SESSION_DIR` unset in the hosted environment; Foundry provides
isolated session storage. Local `.env`, `.azure`, virtual environments, and
generated transcripts are excluded from Git.

## Scope

This sample processes uploaded recordings with **Fast Transcription**; it does
not implement live meeting capture or asynchronous batch processing. See the
[Speech guide](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/fast-transcription-create)
and [hosted-session documentation](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/manage-hosted-sessions)
for supported languages, media requirements, and service limits.

Diarization distinguishes speakers, not their identities or meeting locations.
Use recordings with appropriate authorization, evaluate with representative
audio, and review speaker attribution and generated summaries before sharing.

## References

- [Fast Transcription REST API reference](https://learn.microsoft.com/en-us/rest/api/speechtotext/transcriptions/transcribe?view=rest-speechtotext-2025-10-15)
- [Language identification and diarization](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/configure-language-identification-diarization)
- [Microsoft Foundry samples](https://github.com/microsoft-foundry/foundry-samples)
- [Agent Framework response middleware sample](https://github.com/microsoft/agent-framework/blob/python-1.17.0/python/samples/02-agents/middleware/override_result_with_middleware.py)
