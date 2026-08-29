"""Collect reproducibility metadata without adding benchmark dependencies."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def _command_output(command: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip()
    return output or None


def _cpu_model() -> str:
    cpu_model = platform.processor().strip()
    if cpu_model:
        return cpu_model
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return "unknown"


def collect_environment(torch_module: Any, device: Any) -> Dict[str, Any]:
    """Return the environment fields required by the repository run policy."""

    disk = shutil.disk_usage(Path.cwd())
    cuda_available = bool(torch_module.cuda.is_available())
    gpu: Dict[str, Any] = {
        "available": cuda_available,
        "name": None,
        "driver": None,
        "cuda_runtime": torch_module.version.cuda,
    }
    if device.type == "cuda" and cuda_available:
        gpu["name"] = torch_module.cuda.get_device_name(device)
        gpu["driver"] = _command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                f"--id={device.index or 0}",
            ]
        )

    return {
        "cpu": _cpu_model(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch_module.__version__,
        "gpu": gpu,
        "disk": {
            "path": os.fspath(Path.cwd()),
            "total_bytes": disk.total,
            "free_bytes": disk.free,
        },
    }


def collect_gpu_state(device: Any) -> Optional[Dict[str, Any]]:
    """Sample clock, power and throttle state at the moment of measurement."""
    if getattr(device, "type", None) != "cuda":
        return None

    fields = [
        "clocks.sm",
        "clocks.mem",
        "power.draw",
        "power.limit",
        "temperature.gpu",
        "clocks_throttle_reasons.active",
    ]
    output = _command_output(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
            f"--id={device.index or 0}",
        ]
    )
    if not output:
        return None

    values = [value.strip() for value in output.split(",")]
    if len(values) != len(fields):
        return None
    return dict(zip(fields, values))


def collect_git() -> Dict[str, Any]:
    commit = _command_output(["git", "rev-parse", "HEAD"])
    status = _command_output(["git", "status", "--short"])
    return {"commit": commit, "dirty": bool(status), "status": status or ""}
