import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr

from voice_worker import VoiceWorker


class FakeRuntime:
    def __init__(self, f0_condition=False):
        self.f0_condition = f0_condition
        self.device_name = "cpu"
        self.load_calls = 0
        self.convert_calls = 0

    def load(self):
        self.load_calls += 1

    def convert(self, args):
        self.convert_calls += 1
        with open(args.output_file, "wb") as output_file:
            output_file.write(b"fake wave")
        return args.output_file


class VoiceWorkerTests(unittest.TestCase):
    def test_worker_loads_once_and_completes_conversion_before_shutdown(self):
        runtime = FakeRuntime()
        protocol_output = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = self._create_audio_file(temporary_directory, "source.wav")
            reference = self._create_audio_file(temporary_directory, "reference.wav")
            output = os.path.join(temporary_directory, "result.wav")
            requests = [
                {
                    "id": "convert-1",
                    "command": "convert",
                    "source": source,
                    "reference": reference,
                    "output": output,
                    "diffusionSteps": 4,
                },
                {"id": "ping-1", "command": "ping"},
                {"id": "shutdown-1", "command": "shutdown"},
            ]
            protocol_input = io.StringIO("".join(json.dumps(request) + "\n" for request in requests))
            worker = VoiceWorker(protocol_output, lambda _: runtime)

            exit_code = worker.run(protocol_input)

            messages = self._read_messages(protocol_output)
            self.assertEqual(exit_code, 0)
            self.assertEqual(runtime.load_calls, 1)
            self.assertEqual(runtime.convert_calls, 1)
            self.assertEqual([message["event"] for message in messages], ["ready", "started", "completed", "pong", "shuttingDown"])
            self.assertEqual(messages[2]["output"], os.path.abspath(output))
            self.assertTrue(os.path.isfile(output))
            self.assertFalse(any(file_name.endswith(".partial.wav") for file_name in os.listdir(temporary_directory)))

    def test_invalid_request_reports_error_and_worker_accepts_shutdown(self):
        runtime = FakeRuntime()
        protocol_output = io.StringIO()
        requests = [
            {"id": "convert-1", "command": "convert", "source": "missing.wav"},
            {"id": "shutdown-1", "command": "shutdown"},
        ]
        protocol_input = io.StringIO("".join(json.dumps(request) + "\n" for request in requests))
        worker = VoiceWorker(protocol_output, lambda _: runtime)

        with redirect_stderr(io.StringIO()):
            exit_code = worker.run(protocol_input)

        messages = self._read_messages(protocol_output)
        self.assertEqual(exit_code, 0)
        self.assertEqual([message["event"] for message in messages], ["ready", "error", "shuttingDown"])
        self.assertEqual(messages[1]["id"], "convert-1")
        self.assertEqual(runtime.convert_calls, 0)

    def test_model_load_failure_is_fatal(self):
        protocol_output = io.StringIO()

        def create_failed_runtime(_):
            raise RuntimeError("model load failed")

        worker = VoiceWorker(protocol_output, create_failed_runtime)
        with redirect_stderr(io.StringIO()):
            exit_code = worker.run(io.StringIO())

        messages = self._read_messages(protocol_output)
        self.assertEqual(exit_code, 1)
        self.assertEqual(messages[0]["event"], "fatal")
        self.assertEqual(messages[0]["errorType"], "RuntimeError")

    @staticmethod
    def _create_audio_file(directory, file_name):
        path = os.path.join(directory, file_name)
        with open(path, "wb") as audio_file:
            audio_file.write(b"input")
        return path

    @staticmethod
    def _read_messages(protocol_output):
        return [json.loads(line) for line in protocol_output.getvalue().splitlines()]


if __name__ == "__main__":
    unittest.main()
