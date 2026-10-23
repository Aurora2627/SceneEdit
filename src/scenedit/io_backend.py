"""Explicit safetensors positional reads for constrained-memory Windows hosts."""

import functools
import importlib
import inspect
import safetensors


def install_pread_backend():
    if "backend" not in inspect.signature(safetensors.safe_open).parameters:
        raise RuntimeError("SceneEdit requires safetensors with the pread backend")
    reader = functools.partial(safetensors.safe_open, backend="pread")
    modules = ("diffsynth.core.vram.disk_map", "diffsynth.core.loader.file")
    for name in modules:
        module = importlib.import_module(name)
        if not hasattr(module, "safe_open"):
            raise RuntimeError("Pinned engine reader interface changed: " + name)
        module.safe_open = reader
    return {
        "safetensors": safetensors.__version__,
        "backend": "pread",
        "patched_modules": list(modules),
    }
