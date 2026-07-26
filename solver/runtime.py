"""Offline runtime bootstrap for Qwen3 under the Space's older transformers."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

RUNTIME_PACKAGES = {
    "transformers": "4.51.3",
    "tokenizers": "0.21.1",
    "huggingface_hub": "0.30.2",
    "autoawq": "0.2.9",
}
RUNTIME_WHEELS = (
    "transformers-4.51.3-py3-none-any.whl",
    "tokenizers-0.21.1-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "huggingface_hub-0.30.2-py3-none-any.whl",
    "autoawq-0.2.9-py3-none-any.whl",
)


def _wheelhouse() -> Path:
    env = os.environ.get("IOL_WHEELHOUSE") or os.environ.get("QWEN3_WHEELHOUSE")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "wheelhouse"


def _runtime_dir() -> Path:
    return Path(os.environ.get("IOL_RUNTIME_DIR", "/tmp/iol_qwen3_runtime"))


def installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in RUNTIME_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def ensure_runtime() -> dict[str, str]:
    """Install bundled wheels into /tmp and prefer them on sys.path.

    No-op when wheelhouse is absent (local Qwen2.5 packs / unit tests).
    """
    wheelhouse = _wheelhouse()
    if not wheelhouse.is_dir():
        return installed_versions()

    wheel_paths = [wheelhouse / name for name in RUNTIME_WHEELS]
    missing = [str(path) for path in wheel_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing offline runtime wheels: {missing}")

    runtime_dir = _runtime_dir()
    marker = runtime_dir / ".iol-qwen3-runtime-v1"
    if not marker.is_file():
        runtime_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                "--upgrade",
                "--target",
                str(runtime_dir),
                *(str(path) for path in wheel_paths),
            ],
            check=True,
            timeout=180,
        )
        marker.write_text("offline Qwen3 runtime installed\n", encoding="utf-8")

    runtime_path = str(runtime_dir)
    if runtime_path in sys.path:
        sys.path.remove(runtime_path)
    sys.path.insert(0, runtime_path)
    importlib.invalidate_caches()

    versions = installed_versions()
    mismatches = {
        name: (versions[name], expected)
        for name, expected in RUNTIME_PACKAGES.items()
        if versions[name] != expected
    }
    if mismatches:
        raise RuntimeError(f"Offline runtime version mismatch: {mismatches}")
    print(f"offline runtime: {versions}", flush=True)
    return versions
