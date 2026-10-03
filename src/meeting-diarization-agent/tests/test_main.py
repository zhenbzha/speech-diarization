import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from agent_framework import (
    AgentContext, AgentResponse, AgentResponseUpdate, Content, Message, ResponseStream,
)
from azure.core.exceptions import ClientAuthenticationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from recording_tools import MeetingProcessingError, inspect_transcript, save_meeting_report


class ToolErrorTests(unittest.TestCase):
    def test_success_preserves_tool_result(self):
        result = '{"status": "ok", "evidence": ["test"]}'
        self.assertEqual(main._run_tool(lambda: result), result)

    def test_expected_errors_are_actionable_and_not_success_shaped(self):
        cases = [
            (ValueError("Choose data/first.mp4 or data/second.mp4"), "invalid_input", "data/first.mp4"),
            (MeetingProcessingError("speech_service_error", "HTTP 403; request ID: abc"),
             "speech_service_error", "abc"),
            (ClientAuthenticationError("sensitive detail"), "authentication_error", "RBAC"),
            (httpx.ReadTimeout("sensitive detail"), "speech_connection_error", "timed out"),
            (OSError("sensitive detail"), "file_error", "No download is confirmed"),
        ]
        for exception, code, message in cases:
            def fail():
                raise exception

            with self.subTest(code=code), self.assertLogs("main", level="WARNING") as logs:
                result = json.loads(main._run_tool(fail))
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["error"]["code"], code)
            self.assertIn(message, result["error"]["message"])
            self.assertNotIn("sensitive detail", json.dumps(result) + str(logs.output))

    def test_unexpected_errors_are_logged_and_reraised(self):
        def fail():
            raise RuntimeError("unexpected failure")

        with self.assertLogs("main", level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "unexpected failure"):
                main._run_tool(fail)


class AgentSetupTests(unittest.TestCase):
    def test_agent_registers_persistence_middleware(self):
        with patch.dict(os.environ, {
            "FOUNDRY_PROJECT_ENDPOINT": "https://example.services.ai.azure.com/api/projects/test",
            "AZURE_AI_MODEL_DEPLOYMENT_NAME": "test-model",
        }), patch("main.FoundryChatClient"), patch("main.Agent") as agent:
            main.create_agent(None)
        self.assertEqual(
            agent.call_args.kwargs["middleware"], [main.persist_response_middleware],
        )
        self.assertEqual(
            [tool.name for tool in agent.call_args.kwargs["tools"]],
            ["analyze_meeting_recording", "inspect_diarized_transcript", "get_meeting_files"],
        )
        self.assertEqual(
            agent.call_args.kwargs["default_options"],
            {"store": False, "tool_choice": "required"},
        )

    def test_missing_configuration_is_reported_before_client_creation(self):
        with patch.dict(os.environ, {
            "FOUNDRY_PROJECT_ENDPOINT": "", "AZURE_AI_MODEL_DEPLOYMENT_NAME": "",
        }), patch("main.FoundryChatClient") as client:
            with self.assertRaisesRegex(ValueError, "Missing required configuration"):
                main.create_agent(None)
        client.assert_not_called()

    def test_server_shutdown_closes_owned_credential(self):
        with patch("main.DefaultAzureCredential") as credential, patch(
            "main.create_agent"
        ) as create, patch("main.ResponsesHostServer") as server:
            main.main()
        create.assert_called_once_with(credential.return_value.__enter__.return_value)
        server.return_value.run.assert_called_once()
        credential.return_value.__exit__.assert_called_once()


class RegisteredToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_validation_errors_survive_framework_result_serialization(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"MEETING_SESSION_DIR": directory}
        ), self.assertLogs("main", level="WARNING"):
            contents = await main.analyze_meeting_recording.invoke(arguments={})
        self.assertEqual(len(contents), 1)
        result = json.loads(contents[0].text)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "invalid_input")
        self.assertIn("Upload it", result["error"]["message"])
        self.assertNotIn("Function failed", contents[0].text)

    async def test_transcript_report_and_download_flow(self):
        transcript = json.dumps({"utterances": [{
            "speaker_id": "Speaker 1", "start_ms": 0, "end_ms": 1000,
            "text": "Pilot approved.",
        }]})
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"MEETING_SESSION_DIR": directory}
        ):
            inspected = await main.inspect_diarized_transcript.invoke(
                arguments={"transcript_json": transcript}
            )
            evidence = json.loads(inspected[0].text)
            report = "# Summary\n\nPilot approved. [U1 00:00:00]"
            result = json.loads(save_meeting_report(
                report, evidence["meeting"]["transcript_file"],
                evidence["meeting"]["transcript_sha256"],
            ))
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["report_markdown"], report)
            self.assertTrue(Path(result["summary_file"]).read_text().startswith(report))
            reread = await main.get_meeting_files.invoke(
                arguments={"file_path": evidence["meeting"]["transcript_file"]}
            )
            self.assertEqual(
                json.loads(reread[0].text)["meeting"]["summary_file"],
                result["summary_file"],
            )


class ResponsePersistenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        environment = patch.dict(os.environ, {"MEETING_SESSION_DIR": directory.name})
        environment.start()
        self.addCleanup(environment.stop)
        self.evidence = json.loads(inspect_transcript(json.dumps({"utterances": [{
            "speaker_id": "Speaker 1", "start_ms": 0, "end_ms": 1000,
            "text": "Pilot approved.",
        }]})))
        self.meeting = self.evidence["meeting"]
        self.report = "# Summary\n\nPilot approved. [U1 00:00:00]"
        transcript = Path(self.meeting["transcript_file"])
        self.report_path = transcript.with_name(
            transcript.name.removesuffix(".transcript.json") + ".summary.md"
        )

    def response(self, *, tool_name="analyze_meeting_recording", evidence=None, answer=None):
        messages = [
            Message(role="assistant", contents=[
                Content.from_function_call("ingest", tool_name, arguments="{}"),
            ]),
            Message(role="tool", contents=[
                Content.from_function_result(
                    "ingest", result=json.dumps(self.evidence if evidence is None else evidence),
                ),
            ]),
        ]
        messages.append(Message(
            role="assistant", contents=[self.report if answer is None else answer],
        ))
        return AgentResponse(
            messages=messages, response_id="response-1", finish_reason="stop",
            usage_details={"input_token_count": 10, "output_token_count": 2,
                           "total_token_count": 12},
        )

    async def process(self, response, *, streaming=False):
        context = AgentContext(agent=Mock(), messages=[], stream=streaming)

        async def updates():
            for index, message in enumerate(response.messages):
                yield AgentResponseUpdate(
                    contents=message.contents, role=message.role,
                    message_id=f"message-{index}", response_id=response.response_id,
                )

        async def call_next():
            context.result = (
                ResponseStream(updates(), finalizer=lambda _: response)
                if streaming else response
            )

        await main.persist_response_middleware(context, call_next)
        return context.result

    async def test_omitted_report_tool_is_saved_without_another_model_call(self):
        response = await self.process(self.response())
        persisted = self.report_path.read_text(encoding="utf-8")
        self.assertTrue(persisted.startswith(self.report))
        self.assertIn(self.meeting["transcript_sha256"], persisted)
        self.assertTrue(response.text.startswith(self.report))
        self.assertIn(self.report_path.name, response.text)
        self.assertIn("Playground Files tab", response.text)
        self.assertEqual(response.text.count("# Summary"), 1)

    async def test_stream_emits_nothing_until_report_is_saved(self):
        original = self.response()
        stream = await self.process(original, streaming=True)
        updates = []
        async for update in stream:
            self.assertTrue(self.report_path.is_file())
            updates.append(update)
        final = await stream.get_final_response()
        self.assertTrue(final.text.startswith(self.report))
        self.assertEqual(final.response_id, "response-1")
        self.assertEqual(final.usage_details, original.usage_details)
        self.assertEqual(final.finish_reason, "stop")
        kinds = [content.type for update in updates for content in update.contents]
        self.assertIn("function_call", kinds)
        self.assertIn("function_result", kinds)

    async def test_id_only_citations_are_rendered_from_authoritative_evidence(self):
        response = await self.process(self.response(
            answer=self.report.replace("[U1 00:00:00]", "[U1]"),
        ))
        self.assertTrue(response.text.startswith(self.report))
        self.assertTrue(self.report_path.read_text(encoding="utf-8").startswith(self.report))
        self.assertEqual(response.text.count("# Summary"), 1)

    async def test_new_pasted_transcript_report_is_also_saved(self):
        response = await self.process(self.response(tool_name="inspect_diarized_transcript"))
        self.assertTrue(self.report_path.is_file())
        self.assertTrue(response.text.startswith(self.report))

    async def test_transcript_followup_preserves_existing_report(self):
        save_meeting_report(
            self.report, self.meeting["transcript_file"], self.meeting["transcript_sha256"],
        )
        before = self.report_path.read_bytes()
        evidence = json.loads(inspect_transcript(file_path=self.meeting["transcript_file"]))
        with patch("main.save_meeting_report") as save:
            response = await self.process(self.response(
                tool_name="get_meeting_files", evidence=evidence,
                answer="Here is your transcript.",
            ))
        save.assert_not_called()
        self.assertEqual(self.report_path.read_bytes(), before)
        self.assertIn("Transcript confirmed", response.text)
        self.assertIn(self.report_path.name, response.text)

    async def test_download_followup_does_not_invent_missing_summary(self):
        response = await self.process(self.response(
            tool_name="get_meeting_files", answer="Your summary is ready.",
        ))
        self.assertIn("No saved summary is confirmed", response.text)
        self.assertNotIn("Your summary is ready", response.text)
        self.assertFalse(self.report_path.exists())

    async def test_new_analysis_of_existing_transcript_replaces_report(self):
        save_meeting_report(
            self.report, self.meeting["transcript_file"], self.meeting["transcript_sha256"],
        )
        evidence = json.loads(inspect_transcript(file_path=self.meeting["transcript_file"]))
        response = await self.process(self.response(
            tool_name="inspect_diarized_transcript", evidence=evidence,
            answer="# New analysis\n\nPilot approved. [U1]",
        ))
        self.assertTrue(response.text.startswith("# New analysis"))
        self.assertTrue(
            self.report_path.read_text(encoding="utf-8").startswith("# New analysis")
        )

    async def test_invalid_citations_do_not_escape_stream_or_create_report(self):
        with self.assertLogs("main", level="WARNING"):
            stream = await self.process(self.response(
                answer="Unsupported claim. [U999 00:00:00]",
            ), streaming=True)
            updates = [update async for update in stream]
        text = "".join(update.text for update in updates)
        self.assertIn("Report saving failed (invalid_input)", text)
        self.assertNotIn("Unsupported claim", text)
        self.assertFalse(self.report_path.exists())

    async def test_incorrect_explicit_timestamp_is_rejected_not_silently_repaired(self):
        with self.assertLogs("main", level="WARNING"):
            response = await self.process(self.response(
                answer="Pilot approved. [U1 00:00:09]",
            ))
        self.assertIn("Report saving failed (invalid_input)", response.text)
        self.assertIn("Invalid references: '[U1 00:00:09]'", response.text)
        self.assertFalse(self.report_path.exists())

    async def test_unknown_evidence_id_cannot_create_report(self):
        with self.assertLogs("main", level="WARNING"):
            response = await self.process(self.response(answer="Pilot cancelled. [U999]"))
        self.assertIn("Report saving failed (invalid_input)", response.text)
        self.assertIn("Invalid references: '[U999]'", response.text)
        self.assertNotIn("Pilot cancelled", response.text)
        self.assertFalse(self.report_path.exists())

    async def test_changed_transcript_is_not_bound_to_stale_evidence(self):
        path = Path(self.meeting["transcript_file"])
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertLogs("main", level="WARNING"):
            response = await self.process(self.response())
        self.assertIn("transcript changed", response.text)
        self.assertFalse(self.report_path.exists())

    async def test_write_failure_replaces_success_claim(self):
        with patch("main.save_meeting_report", side_effect=OSError("disk full")), \
                self.assertLogs("main", level="WARNING"):
            response = await self.process(self.response())
        self.assertIn("Report saving failed (file_error)", response.text)
        self.assertNotIn("# Summary", response.text)
        self.assertFalse(self.report_path.exists())

    async def test_ingestion_failure_blocks_invented_analysis(self):
        error = {"status": "error", "error": {
            "code": "invalid_input", "message": "Upload a meeting recording.",
        }}
        response = await self.process(self.response(
            evidence=error, answer="Invented summary is available.",
        ))
        self.assertIn("Upload a meeting recording", response.text)
        self.assertNotIn("Invented summary", response.text)
        self.assertFalse(self.report_path.exists())

    async def test_incomplete_response_does_not_create_report(self):
        response = self.response()
        response.finish_reason = "length"
        with self.assertLogs("main", level="WARNING"):
            result = await self.process(response)
        self.assertIn("response was incomplete", result.text)
        self.assertFalse(self.report_path.exists())

    async def test_response_without_ingestion_is_withheld(self):
        response = AgentResponse(messages=[
            Message(role="assistant", contents=["Unsaved meeting analysis. [U1]"]),
        ])
        with self.assertLogs("main", level="WARNING"):
            result = await self.process(response)
        self.assertIn("No validated meeting evidence", result.text)
        self.assertNotIn("Unsaved meeting analysis", result.text)
        self.assertFalse(self.report_path.exists())


if __name__ == "__main__":
    unittest.main()
