"""Persistent JSON-lines process boundary for Seed-VC inference."""

import argparse
import json
import os
import sys
import time
import traceback
import uuid


PROTOCOL_VERSION = 1


class VoiceWorker:
    def __init__(self, protocol_output, runtime_factory, f0_condition=False):
        self.protocol_output = protocol_output
        self.runtime_factory = runtime_factory
        self.f0_condition = bool(f0_condition)
        self.runtime = None

    def run(self, protocol_input):
        try:
            self.runtime = self.runtime_factory(self.f0_condition)
            self.runtime.load()
            self._emit(
                {
                    "event": "ready",
                    "protocolVersion": PROTOCOL_VERSION,
                    "device": self.runtime.device_name,
                    "f0Condition": self.f0_condition,
                }
            )
        except Exception as error:
            self._emit_error(None, error, event="fatal")
            traceback.print_exc(file=sys.stderr)
            return 1

        for line in protocol_input:
            if not line.strip():
                continue

            should_continue = self._handle_line(line)
            if not should_continue:
                return 0

        return 0

    def _handle_line(self, line):
        try:
            request = json.loads(line)
        except (TypeError, json.JSONDecodeError) as error:
            self._emit_error(None, ValueError(f"Invalid JSON request: {error}"))
            return True

        if not isinstance(request, dict):
            self._emit_error(None, ValueError("Each request must be a JSON object."))
            return True

        request_id = request.get("id")
        if not isinstance(request_id, str) or not request_id.strip():
            self._emit_error(None, ValueError("Each request requires a non-empty string id."))
            return True

        command = request.get("command")
        if command == "ping":
            self._emit({"id": request_id, "event": "pong", "device": self.runtime.device_name})
            return True
        if command == "shutdown":
            self._emit({"id": request_id, "event": "shuttingDown"})
            return False
        if command != "convert":
            self._emit_error(request_id, ValueError(f"Unsupported command: {command!r}"))
            return True

        self._convert(request_id, request)
        return True

    def _convert(self, request_id, request):
        temporary_output = None
        try:
            source = self._require_existing_file(request, "source")
            reference = self._require_existing_file(request, "reference")
            output = self._require_output_path(request)
            diffusion_steps = self._read_int(request, "diffusionSteps", 25, 1, 200)
            length_adjust = self._read_float(request, "lengthAdjust", 1.0, 0.25, 4.0)
            inference_cfg_rate = self._read_float(request, "inferenceCfgRate", 0.7, 0.0, 5.0)
            semi_tone_shift = self._read_int(request, "semiToneShift", 0, -48, 48)
            auto_f0_adjust = self._read_bool(request, "autoF0Adjust", True)

            os.makedirs(os.path.dirname(output), exist_ok=True)
            temporary_output = f"{output}.{uuid.uuid4().hex}.partial.wav"
            inference_args = argparse.Namespace(
                source=source,
                target=reference,
                output=os.path.dirname(temporary_output),
                output_file=temporary_output,
                diffusion_steps=diffusion_steps,
                length_adjust=length_adjust,
                inference_cfg_rate=inference_cfg_rate,
                f0_condition=self.f0_condition,
                auto_f0_adjust=auto_f0_adjust,
                semi_tone_shift=semi_tone_shift,
            )

            self._emit({"id": request_id, "event": "started", "device": self.runtime.device_name})
            started_at = time.monotonic()
            self.runtime.convert(inference_args)
            os.replace(temporary_output, output)
            temporary_output = None
            self._emit(
                {
                    "id": request_id,
                    "event": "completed",
                    "device": self.runtime.device_name,
                    "output": output,
                    "durationSeconds": round(time.monotonic() - started_at, 3),
                }
            )
        except Exception as error:
            if temporary_output and os.path.exists(temporary_output):
                try:
                    os.remove(temporary_output)
                except OSError:
                    traceback.print_exc(file=sys.stderr)
            self._emit_error(request_id, error)
            traceback.print_exc(file=sys.stderr)

    def _emit(self, message):
        self.protocol_output.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.protocol_output.flush()

    def _emit_error(self, request_id, error, event="error"):
        self._emit(
            {
                "id": request_id,
                "event": event,
                "errorType": type(error).__name__,
                "error": str(error),
            }
        )

    @staticmethod
    def _require_existing_file(request, field_name):
        value = request.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'{field_name}' must be a non-empty path string.")

        path = os.path.abspath(os.path.expanduser(value))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{field_name.capitalize()} audio file does not exist: {path}")
        return path

    @staticmethod
    def _require_output_path(request):
        value = request.get("output")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("'output' must be a non-empty path string.")

        path = os.path.abspath(os.path.expanduser(value))
        if os.path.splitext(path)[1].lower() != ".wav":
            raise ValueError("'output' must use the .wav extension.")
        return path

    @staticmethod
    def _read_int(request, field_name, default, minimum, maximum):
        value = request.get(field_name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"'{field_name}' must be an integer.")
        if value < minimum or value > maximum:
            raise ValueError(f"'{field_name}' must be between {minimum} and {maximum}.")
        return value

    @staticmethod
    def _read_float(request, field_name, default, minimum, maximum):
        value = request.get(field_name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"'{field_name}' must be a number.")
        if value < minimum or value > maximum:
            raise ValueError(f"'{field_name}' must be between {minimum} and {maximum}.")
        return float(value)

    @staticmethod
    def _read_bool(request, field_name, default):
        value = request.get(field_name, default)
        if not isinstance(value, bool):
            raise ValueError(f"'{field_name}' must be true or false.")
        return value


def create_runtime(f0_condition):
    from inference import SeedVCInferenceRuntime

    return SeedVCInferenceRuntime(f0_condition)


def main():
    parser = argparse.ArgumentParser(description="Run Seed-VC as a persistent JSON-lines worker.")
    parser.add_argument("--f0-condition", action="store_true")
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    worker_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(worker_directory)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.device:
        os.environ["SEED_VC_DEVICE"] = args.device

    packaged_model_directory = os.path.abspath(os.path.join(worker_directory, "..", "models"))
    if os.path.isdir(packaged_model_directory):
        os.environ.setdefault("SEED_VC_MODEL_DIR", packaged_model_directory)
        encoder_directory = os.path.join(packaged_model_directory, "whisper-small-encoder")
    else:
        encoder_directory = os.path.join(worker_directory, "checkpoints", "worker", "whisper-small-encoder")

    if os.path.isdir(encoder_directory):
        os.environ.setdefault("SEED_VC_WHISPER_ENCODER_DIR", encoder_directory)

    # Third-party model loaders use stdout for human-readable diagnostics. The
    # protocol must remain machine-readable, so preserve the original stream for
    # JSON messages and route all later stdout writes to the diagnostic stream.
    protocol_output = sys.stdout
    sys.stdout = sys.stderr
    worker = VoiceWorker(protocol_output, create_runtime, f0_condition=args.f0_condition)
    return worker.run(sys.stdin)


if __name__ == "__main__":
    raise SystemExit(main())
