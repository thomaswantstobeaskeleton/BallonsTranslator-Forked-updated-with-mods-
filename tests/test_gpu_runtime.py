import sys

from utils.gpu_runtime import (
    GPU_PROFILE_AMD_DIRECTML,
    GPU_PROFILE_AMD_ROCM_PREVIEW,
    GPU_PROFILE_CPU,
    GPU_PROFILE_NVIDIA_CUDA,
    build_gpu_install_plan,
    classify_amd_gpu,
    normalize_gpu_profile,
    python_supports_pinned_cuda_torch,
)


def _win_py312():
    return type("Version", (), {"major": 3, "minor": 12})()


def _win_py310():
    return type("Version", (), {"major": 3, "minor": 10})()


def _win_py314():
    return type("Version", (), {"major": 3, "minor": 14})()


def test_normalize_gpu_profile_aliases():
    assert normalize_gpu_profile("cuda") == GPU_PROFILE_NVIDIA_CUDA
    assert normalize_gpu_profile("rocm") == GPU_PROFILE_AMD_ROCM_PREVIEW
    assert normalize_gpu_profile("dml") == GPU_PROFILE_AMD_DIRECTML
    assert normalize_gpu_profile("off") == GPU_PROFILE_CPU


def test_auto_prefers_nvidia_cuda_when_nvidia_present():
    plan = build_gpu_install_plan("auto", python_version_info=_win_py312(), gpu_names=["NVIDIA GeForce RTX 4070"])
    assert plan.resolved_profile == GPU_PROFILE_NVIDIA_CUDA
    assert "cu118" in plan.torch_command


def test_rx9070_auto_uses_rocm_preview_on_windows_python312(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    plan = build_gpu_install_plan("auto", python_version_info=_win_py312(), gpu_names=["AMD Radeon RX 9070 XT"])
    assert classify_amd_gpu(plan.physical_gpus) == "RDNA4"
    assert plan.resolved_profile == GPU_PROFILE_AMD_ROCM_PREVIEW
    assert "repo.radeon.com/rocm/windows" in plan.torch_command


def test_rx9070_auto_falls_back_to_directml_without_python312(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    plan = build_gpu_install_plan("auto", python_version_info=_win_py310(), gpu_names=["AMD Radeon RX 9070 XT"])
    assert plan.resolved_profile == GPU_PROFILE_AMD_DIRECTML
    assert "torch-directml" in plan.torch_command
    assert plan.warnings


def test_cuda_command_drops_version_pin_on_unsupported_python():
    plan = build_gpu_install_plan("auto", python_version_info=_win_py314(), gpu_names=["NVIDIA GeForce RTX 3070 Ti"])
    assert plan.resolved_profile == GPU_PROFILE_NVIDIA_CUDA
    assert "torch==2.7.1" not in plan.torch_command
    assert "cu118" not in plan.torch_command
    assert plan.warnings


def test_cuda_command_keeps_pin_on_supported_python():
    assert python_supports_pinned_cuda_torch(_win_py312())
    assert not python_supports_pinned_cuda_torch(_win_py314())
    plan = build_gpu_install_plan("cuda", python_version_info=_win_py312(), gpu_names=["NVIDIA GeForce RTX 4070"])
    assert "torch==2.7.1" in plan.torch_command


def test_no_gpu_uses_cpu_profile():
    plan = build_gpu_install_plan("auto", python_version_info=_win_py312(), gpu_names=["Microsoft Basic Display Adapter"])
    assert plan.resolved_profile == GPU_PROFILE_CPU
    assert "whl/cpu" in plan.torch_command
