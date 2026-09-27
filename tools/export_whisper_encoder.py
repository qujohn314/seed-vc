import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoFeatureExtractor, WhisperModel


def export_encoder(source_model, revision, output_directory, dtype):
    output_path = Path(output_directory).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    model = WhisperModel.from_pretrained(source_model, revision=revision)
    feature_extractor = AutoFeatureExtractor.from_pretrained(source_model, revision=revision)

    target_dtype = torch.float16 if dtype == "float16" else torch.float32
    encoder_state = {
        name: value.detach().to(dtype=target_dtype, device="cpu").contiguous()
        for name, value in model.encoder.state_dict().items()
    }

    save_file(encoder_state, output_path / "encoder.safetensors")
    model.config.save_pretrained(output_path)
    feature_extractor.save_pretrained(output_path)

    metadata = {
        "source_model": source_model,
        "source_revision": revision,
        "dtype": dtype,
        "parameter_count": sum(parameter.numel() for parameter in model.encoder.parameters()),
    }
    (output_path / "encoder_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", default="openai/whisper-small")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    arguments = parser.parse_args()
    export_encoder(arguments.source_model, arguments.revision, arguments.output, arguments.dtype)
