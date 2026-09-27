import argparse
import unittest
from unittest.mock import Mock, patch

import torch

import inference


class SeedVCInferenceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.original_device = inference.device

    def tearDown(self):
        inference.device = self.original_device

    def test_model_load_cuda_failure_reloads_once_on_cpu(self):
        inference.device = torch.device("cuda")
        cpu_models = ("cpu-models",)
        runtime = inference.SeedVCInferenceRuntime(f0_condition=False)

        with patch.object(
            inference,
            "load_models",
            side_effect=[RuntimeError("CUDA driver initialization failed"), cpu_models],
        ) as load_models, patch.object(inference, "release_cuda_memory") as release_cuda_memory:
            runtime.load()

        self.assertEqual(load_models.call_count, 2)
        self.assertEqual(release_cuda_memory.call_count, 1)
        self.assertIs(runtime.models, cpu_models)
        self.assertEqual(runtime.device_name, "cpu")

    def test_conversion_cuda_failure_reloads_models_and_retries_on_cpu(self):
        inference.device = torch.device("cuda")
        gpu_models = ("gpu-models",)
        cpu_models = ("cpu-models",)
        args = argparse.Namespace(f0_condition=False)
        runtime = inference.SeedVCInferenceRuntime(f0_condition=False)
        runtime.models = gpu_models

        convert = Mock(side_effect=[RuntimeError("CUDA out of memory"), "output.wav"])
        with patch.object(inference, "run_inference", convert), patch.object(
            inference,
            "load_models",
            return_value=cpu_models,
        ) as load_models, patch.object(inference, "release_cuda_memory") as release_cuda_memory:
            result = runtime.convert(args)

        self.assertEqual(result, "output.wav")
        self.assertEqual(convert.call_count, 2)
        self.assertIs(convert.call_args_list[0].args[1], gpu_models)
        self.assertIs(convert.call_args_list[1].args[1], cpu_models)
        self.assertEqual(load_models.call_count, 1)
        self.assertEqual(release_cuda_memory.call_count, 1)
        self.assertEqual(runtime.device_name, "cpu")

    def test_non_cuda_error_is_not_retried(self):
        inference.device = torch.device("cuda")
        args = argparse.Namespace(f0_condition=False)
        runtime = inference.SeedVCInferenceRuntime(f0_condition=False)
        runtime.models = ("gpu-models",)

        with patch.object(inference, "run_inference", side_effect=FileNotFoundError("missing input")) as convert, patch.object(
            inference,
            "load_models",
        ) as load_models:
            with self.assertRaises(FileNotFoundError):
                runtime.convert(args)

        self.assertEqual(convert.call_count, 1)
        self.assertEqual(load_models.call_count, 0)
        self.assertEqual(runtime.device_name, "cuda")


if __name__ == "__main__":
    unittest.main()
