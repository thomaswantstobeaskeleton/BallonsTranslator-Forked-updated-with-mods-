from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

GPU_PROFILE_AUTO = "auto"
GPU_PROFILE_CPU = "cpu"
GPU_PROFILE_NVIDIA_CUDA = "nvidia-cuda"
GPU_PROFILE_AMD_DIRECTML = "amd-directml"
GPU_PROFILE_AMD_ROCM_PREVIEW = "amd-rocm-preview"
GPU_PROFILES = {
    GPU_PROFILE_AUTO,
    GPU_PROFILE_CPU,
    GPU_PROFILE_NVIDIA_CUDA,
    GPU_PROFILE_AMD_DIRECTML,
    GPU_PROFILE_AMD_ROCM_PREVIEW,
}

NVIDIA_CUDA_TORCH_COMMAND = (
    "pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 "
    "--index-url https://download.pytorch.org/whl/cu118 --disable-pip-version-check"
)
# The pinned torch 2.7.1 cu118 build only publishes wheels for CPython 3.9-3.13.
# Newer interpreters (e.g. 3.14) get "No matching distribution found for torch==2.7.1"
# from the cu118 index, which used to abort startup (issue #154).  Fall back to the
# newest CUDA index without a version pin so pip can pick whatever build supports
# the running interpreter.
NVIDIA_CUDA_PINNED_PY_MIN = (3, 9)
NVIDIA_CUDA_PINNED_PY_MAX = (3, 13)
NVIDIA_CUDA_LATEST_TORCH_COMMAND = (
    "pip install torch torchvision torchaudio "
    "--index-url https://download.pytorch.org/whl/cu128 --disable-pip-version-check"
)
CPU_TORCH_COMMAND = (
    "pip install torch torchvision torchaudio "
    "--index-url https://download.pytorch.org/whl/cpu --disable-pip-version-check"
)
AMD_DIRECTML_TORCH_COMMAND = "pip install torch-directml --disable-pip-version-check"
# AMD's current Windows ROCm preview wheels target CPython 3.12.  Keep the URL
# centralized so launch.py, batch files, docs, and tests agree.
AMD_ROCM_PREVIEW_TORCH_COMMAND = (
    "pip install "
    "https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torch-2.8.0a0%2Bgitfc14c65-cp312-cp312-win_amd64.whl "
    "https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torchvision-0.24.0a0%2Bc85f008-cp312-cp312-win_amd64.whl "
    "https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torchaudio-2.6.0a0%2B1a8f621-cp312-cp312-win_amd64.whl "
    "--disable-pip-version-check"
)

AMD_RDNA3_PATTERNS = (
    "RX 7900", "RX 7800", "RX 7700", "RX 7600",
    "PRO W7900", "PRO W7800", "PRO W7700", "PRO W7600",
)
AMD_RDNA4_PATTERNS = (
    "RX 9070", "RX 9060", "RX 9050", "AI PRO R9700",
)


@dataclass(frozen=True)
class GpuInstallPlan:
    requested_profile: str
    resolved_profile: str
    torch_command: str
    required_packages: tuple[str, ...]
    reason: str
    warnings: tuple[str, ...] = ()
    physical_gpus: tuple[str, ...] = ()
    amd_family: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "requested_profile": self.requested_profile,
            "resolved_profile": self.resolved_profile,
            "torch_command": self.torch_command,
            "required_packages": list(self.required_packages),
            "reason": self.reason,
            "warnings": list(self.warnings),
            "physical_gpus": list(self.physical_gpus),
            "amd_family": self.amd_family,
        }


def normalize_gpu_profile(profile: Optional[str]) -> str:
    value = (profile or GPU_PROFILE_AUTO).strip().lower().replace("_", "-")
    aliases = {
        "nvidia": GPU_PROFILE_NVIDIA_CUDA,
        "cuda": GPU_PROFILE_NVIDIA_CUDA,
        "amd": GPU_PROFILE_AMD_DIRECTML,
        "directml": GPU_PROFILE_AMD_DIRECTML,
        "dml": GPU_PROFILE_AMD_DIRECTML,
        "rocm": GPU_PROFILE_AMD_ROCM_PREVIEW,
        "amd-rocm": GPU_PROFILE_AMD_ROCM_PREVIEW,
        "nightly": GPU_PROFILE_AMD_ROCM_PREVIEW,
        "amd-nightly": GPU_PROFILE_AMD_ROCM_PREVIEW,
        "off": GPU_PROFILE_CPU,
        "none": GPU_PROFILE_CPU,
    }
    value = aliases.get(value, value)
    return value if value in GPU_PROFILES else GPU_PROFILE_AUTO


def _run_text(cmd: List[str], timeout: int = 8) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=timeout)
    except Exception:
        return ""


def _windows_gpu_names() -> List[str]:
    if sys.platform != "win32":
        return []
    outputs = []
    outputs.append(_run_text(["wmic", "path", "win32_VideoController", "get", "name"]))
    ps = _run_text([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
    ])
    outputs.append(ps)
    names: List[str] = []
    for output in outputs:
        for line in (output or "").splitlines():
            line = line.strip()
            if not line or line.lower() == "name":
                continue
            if line not in names:
                names.append(line)
    return names


def detect_physical_gpus() -> List[str]:
    names = []
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        nvsmi = _run_text(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        for line in nvsmi.splitlines():
            line = line.strip()
            if line and line not in names:
                names.append(line)
    for line in _windows_gpu_names():
        if line not in names:
            names.append(line)
    return names


def has_nvidia_gpu(names: Optional[Iterable[str]] = None) -> bool:
    haystack = "\n".join(names if names is not None else detect_physical_gpus()).lower()
    return any(k in haystack for k in ("nvidia", "geforce", "rtx", "gtx", "quadro"))


def has_amd_gpu(names: Optional[Iterable[str]] = None) -> bool:
    haystack = "\n".join(names if names is not None else detect_physical_gpus()).lower()
    return any(k in haystack for k in ("amd", "radeon", "rx ", "pro w"))


def classify_amd_gpu(names: Optional[Iterable[str]] = None) -> str:
    text = "\n".join(names if names is not None else detect_physical_gpus()).upper()
    if any(pattern in text for pattern in AMD_RDNA4_PATTERNS):
        return "RDNA4"
    if any(pattern in text for pattern in AMD_RDNA3_PATTERNS):
        return "RDNA3"
    if "RADEON" in text or "AMD" in text:
        return "AMD"
    return ""


def python_supports_pinned_cuda_torch(version_info=None) -> bool:
    """True when the pinned CUDA (cu118) wheels publish a build for this interpreter."""
    vi = version_info or sys.version_info
    version = (int(vi.major), int(vi.minor))
    return NVIDIA_CUDA_PINNED_PY_MIN <= version <= NVIDIA_CUDA_PINNED_PY_MAX


def nvidia_cuda_torch_command(version_info=None) -> str:
    """CUDA torch install command that matches the running interpreter."""
    if python_supports_pinned_cuda_torch(version_info):
        return NVIDIA_CUDA_TORCH_COMMAND
    return NVIDIA_CUDA_LATEST_TORCH_COMMAND


def python_supports_amd_rocm_preview(version_info=None) -> bool:
    vi = version_info or sys.version_info
    return sys.platform == "win32" and int(vi.major) == 3 and int(vi.minor) == 12


def build_gpu_install_plan(profile: Optional[str] = None, *, nightly: bool = False, python_version_info=None, gpu_names: Optional[Iterable[str]] = None) -> GpuInstallPlan:
    requested = GPU_PROFILE_AMD_ROCM_PREVIEW if nightly else normalize_gpu_profile(profile or os.environ.get("BT_GPU_PROFILE") or GPU_PROFILE_AUTO)
    names = tuple(gpu_names if gpu_names is not None else detect_physical_gpus())
    amd_family = classify_amd_gpu(names)
    warnings: List[str] = []

    resolved = requested
    reason = "Explicit GPU profile requested."
    if requested == GPU_PROFILE_AUTO:
        if has_nvidia_gpu(names):
            resolved = GPU_PROFILE_NVIDIA_CUDA
            reason = "NVIDIA GPU detected; using CUDA PyTorch wheels."
        elif has_amd_gpu(names):
            if amd_family in {"RDNA3", "RDNA4"} and python_supports_amd_rocm_preview(python_version_info):
                resolved = GPU_PROFILE_AMD_ROCM_PREVIEW
                reason = f"AMD {amd_family} GPU detected with Python 3.12; using AMD ROCm Windows preview wheels."
            else:
                resolved = GPU_PROFILE_AMD_DIRECTML
                if amd_family in {"RDNA3", "RDNA4"}:
                    reason = f"AMD {amd_family} GPU detected, but ROCm preview wheels require Windows CPython 3.12; using DirectML."
                    warnings.append("For ROCm preview on Radeon RX 7000/9000, use Python 3.12 or set BT_GPU_PROFILE=amd-rocm-preview.")
                else:
                    reason = "AMD GPU detected; using DirectML because it is the broad Windows AMD fallback."
        else:
            resolved = GPU_PROFILE_CPU
            reason = "No NVIDIA/AMD GPU detected; using CPU PyTorch wheels."
    elif requested == GPU_PROFILE_AMD_ROCM_PREVIEW and not python_supports_amd_rocm_preview(python_version_info):
        resolved = GPU_PROFILE_AMD_DIRECTML
        reason = "AMD ROCm preview was requested, but the published Windows wheels target CPython 3.12; falling back to DirectML."
        warnings.append("Install/use Python 3.12 to try AMD ROCm preview wheels on Radeon RX 7000/9000.")

    if resolved == GPU_PROFILE_NVIDIA_CUDA:
        command = nvidia_cuda_torch_command(python_version_info)
        required = ("torch", "torchvision")
        if not python_supports_pinned_cuda_torch(python_version_info):
            vi = python_version_info or sys.version_info
            pinned_max = ".".join(str(p) for p in NVIDIA_CUDA_PINNED_PY_MAX)
            reason += (
                f" Python {int(vi.major)}.{int(vi.minor)} has no cu118 wheels for the pinned torch build;"
                " using the latest CUDA wheels without a version pin."
            )
            warnings.append(
                f"CUDA wheels are pinned only for CPython {'.'.join(str(p) for p in NVIDIA_CUDA_PINNED_PY_MIN)}-{pinned_max}. "
                "If installation fails on this interpreter, install Python 3.12 or set TORCH_COMMAND to a working pip command."
            )
    elif resolved == GPU_PROFILE_AMD_ROCM_PREVIEW:
        command = AMD_ROCM_PREVIEW_TORCH_COMMAND
        required = ("torch", "torchvision")
    elif resolved == GPU_PROFILE_AMD_DIRECTML:
        command = AMD_DIRECTML_TORCH_COMMAND
        required = ("torch", "torch_directml")
    else:
        command = CPU_TORCH_COMMAND
        required = ("torch", "torchvision")

    command = os.environ.get("TORCH_COMMAND", command)
    if "TORCH_COMMAND" in os.environ:
        reason += " TORCH_COMMAND override is set."

    return GpuInstallPlan(
        requested_profile=requested,
        resolved_profile=resolved,
        torch_command=command,
        required_packages=required,
        reason=reason,
        warnings=tuple(warnings),
        physical_gpus=names,
        amd_family=amd_family,
    )


def format_gpu_install_plan(plan: GpuInstallPlan) -> str:
    lines = [
        f"GPU profile: requested={plan.requested_profile}, resolved={plan.resolved_profile}",
        f"Detected physical GPUs: {', '.join(plan.physical_gpus) if plan.physical_gpus else 'none'}",
        f"AMD family: {plan.amd_family or 'n/a'}",
        f"Reason: {plan.reason}",
        f"Torch install command: python -m {plan.torch_command}",
    ]
    for warning in plan.warnings:
        lines.append(f"Warning: {warning}")
    return "\n".join(lines)
