import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gui.threads.yolo_inference_threads import (
    get_model_loading_kwargs,
    resolve_inference_device,
    should_fallback_to_cpu,
)


def test_get_model_loading_kwargs_for_supported_models_force_detect_task():
    assert get_model_loading_kwargs("/tmp/model.pt") == {"task": "detect"}
    assert get_model_loading_kwargs("/tmp/model.onnx") == {"task": "detect"}
    assert get_model_loading_kwargs("/tmp/model.engine") == {"task": "detect"}
    assert get_model_loading_kwargs("/tmp/model.mlmodel") == {"task": "detect"}


def test_get_model_loading_kwargs_for_unsupported_extensions_returns_empty_dict():
    assert get_model_loading_kwargs("/tmp/model.bin") == {}


def test_resolve_inference_device_for_onnx_auto_prefers_cuda():
    assert resolve_inference_device("/tmp/model.onnx", "auto") == "cuda"
    assert resolve_inference_device("/tmp/model.engine", "auto") == "cuda"
    assert resolve_inference_device("/tmp/model.onnx", "0") == "0"
    assert resolve_inference_device("/tmp/model.pt", "auto") is None


def test_should_fallback_to_cpu_for_cuda_provider_errors():
    error = "Failed to load library libonnxruntime_providers_cuda.so with error: libcublasLt.so.11"
    assert should_fallback_to_cpu("/tmp/model.onnx", "auto", error) is True
    assert should_fallback_to_cpu("/tmp/model.pt", "auto", error) is False
    assert should_fallback_to_cpu("/tmp/model.onnx", "cpu", "some other error") is False
