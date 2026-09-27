import os
from huggingface_hub import hf_hub_download


PACKAGED_REPOSITORY_DIRECTORIES = {
    "Plachta/Seed-VC": "seed-vc",
    "funasr/campplus": "campplus",
    "lj1995/VoiceConversionWebUI": "rmvpe",
}


def resolve_packaged_model(repo_id, filename):
    model_root = os.environ.get("SEED_VC_MODEL_DIR")
    if not model_root:
        return None

    repository_directory = PACKAGED_REPOSITORY_DIRECTORIES.get(repo_id)
    if not repository_directory:
        raise ValueError(f"No packaged model directory is configured for repository '{repo_id}'.")

    model_path = os.path.abspath(os.path.join(model_root, repository_directory, filename))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Packaged model file does not exist: {model_path}")
    return model_path


def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename="config.yml"):
    packaged_model_path = resolve_packaged_model(repo_id, model_filename)
    if packaged_model_path:
        if config_filename is None:
            return packaged_model_path
        return packaged_model_path, resolve_packaged_model(repo_id, config_filename)

    os.makedirs("./checkpoints", exist_ok=True)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir="./checkpoints")
    if config_filename is None:
        return model_path
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir="./checkpoints")

    return model_path, config_path
