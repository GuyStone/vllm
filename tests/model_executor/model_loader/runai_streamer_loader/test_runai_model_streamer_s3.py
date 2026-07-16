# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from huggingface_hub import snapshot_download
from runai_model_streamer.safetensors_streamer.streamer_mock import StreamerPatcher

from vllm.engine.arg_utils import EngineArgs

from .conftest import RunaiDummyExecutor

load_format = "runai_streamer"
test_model = "openai-community/gpt2"
mtp_test_model = "luccafong/deepseek_mtp_main_random"


def test_runai_model_loader_download_files_s3_mocked_with_patch(
    vllm_runner,
    tmp_path: Path,
    monkeypatch,
):
    patcher = StreamerPatcher(str(tmp_path))

    test_mock_s3_model = "s3://my-mock-bucket/gpt2/"

    # Download model from HF
    mock_model_dir = f"{tmp_path}/gpt2"
    snapshot_download(repo_id=test_model, local_dir=mock_model_dir)

    monkeypatch.setattr(
        "vllm.transformers_utils.runai_utils.runai_list_safetensors",
        patcher.shim_list_safetensors,
    )
    monkeypatch.setattr(
        "vllm.transformers_utils.runai_utils.runai_pull_files",
        patcher.shim_pull_files,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.weight_utils.SafetensorsStreamer",
        patcher.create_mock_streamer,
    )

    engine_args = EngineArgs(
        model=test_mock_s3_model,
        load_format=load_format,
        tensor_parallel_size=1,
    )

    vllm_config = engine_args.create_engine_config()

    executor = RunaiDummyExecutor(vllm_config)
    executor.driver_worker.load_model()


def test_runai_model_loader_mtp_drafter_streams_from_s3_mocked_with_patch(
    vllm_runner,
    tmp_path: Path,
    monkeypatch,
):
    """The MTP drafter shares the target checkpoint. When the target is
    resolved from object storage, its config is staged into a local cache dir
    that never contains safetensors, so the draft model config must inherit
    the object-storage weight source (``model_weights``) for drafter weights
    to stream from the same source as the target."""
    patcher = StreamerPatcher(str(tmp_path))

    test_mock_s3_model = "s3://my-mock-bucket/deepseek-mtp/"

    mock_model_dir = f"{tmp_path}/deepseek-mtp"
    snapshot_download(repo_id=mtp_test_model, local_dir=mock_model_dir)

    monkeypatch.setattr(
        "vllm.transformers_utils.runai_utils.runai_list_safetensors",
        patcher.shim_list_safetensors,
    )
    monkeypatch.setattr(
        "vllm.transformers_utils.runai_utils.runai_pull_files",
        patcher.shim_pull_files,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.weight_utils.SafetensorsStreamer",
        patcher.create_mock_streamer,
    )

    engine_args = EngineArgs(
        model=test_mock_s3_model,
        load_format=load_format,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=2048,
        gpu_memory_utilization=0.6,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
    )

    vllm_config = engine_args.create_engine_config()

    draft_model_config = vllm_config.speculative_config.draft_model_config
    assert draft_model_config.model_weights == test_mock_s3_model

    executor = RunaiDummyExecutor(vllm_config)
    executor.driver_worker.load_model()
