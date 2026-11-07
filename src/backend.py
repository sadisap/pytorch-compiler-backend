
import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import torch

from .codegen import generate_cpp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = PROJECT_ROOT / "generated"

COMPILATION_RECORDS = []



def compile_cpp(source_path, library_path, opt_level="O0"):
    system = platform.system()

    if system == "Darwin":
        command = [
            "clang++", "-std=c++17", f"-{opt_level}", "-dynamiclib",
        ]
    elif system == "Linux":
        command = [
            "g++", "-std=c++17", f"-{opt_level}", "-shared", "-fPIC",
        ]
    else:
        raise RuntimeError(f"Unsupported operating system: {system}")

    # Compile to a private temporary library first.
    with tempfile.TemporaryDirectory(dir=GENERATED_DIR) as temp_dir:
        temp_library = Path(temp_dir) / library_path.name

        command += [
            str(source_path), "-o", str(temp_library)
        ]

        start = time.perf_counter()
        subprocess.run(
            command, check=True, capture_output=True, text=True
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Publish only after compilation succeeds.
        os.replace(temp_library, library_path)

    return elapsed_ms


def make_cpp_backend(fuse=False, opt_level="O0"):
    if opt_level not in {"O0", "O2", "O3"}:
        raise ValueError(f"Unsupported optimization level: {opt_level}")
    mode = "fused" if fuse else "unfused"

    def cpp_backend(gm, example_inputs):
        print(f"\n=== {mode.upper()} C++ Backend ===")
        print(gm.graph)

        tensor_inputs = [x for x in example_inputs if isinstance(x, torch.Tensor)]

        if len(tensor_inputs) != 1:
            raise NotImplementedError("Exactly one tensor input is supported.")

        example = tensor_inputs[0]

        if example.device.type != "cpu" or example.dtype != torch.float32:
            raise NotImplementedError("Only CPU float32 tensors are supported.")

        GENERATED_DIR.mkdir(parents=True, exist_ok=True)

        source_path = GENERATED_DIR / (
            f"kernel_{mode}_{uuid.uuid4().hex}.cpp"
        )

        extension = ".dylib" if platform.system() == "Darwin" else ".so"

        cpp_code = generate_cpp(gm, source_path, fuse=fuse)

        system = platform.system()
        compiler_name = "clang++" if system == "Darwin" else "g++"
        compiler_path = shutil.which(compiler_name)

        if compiler_path is None:
            raise RuntimeError(f"Compiler not found: {compiler_name}")

        compiler_version = subprocess.check_output(
            [compiler_path, "--version"], text=True
        )

        cache_key = hashlib.sha256(
            (
                cpp_code
                + "|"
                + opt_level
                + "|"
                + system
                + "|"
                + platform.machine()
                + "|"
                + compiler_path
                + "|"
                + compiler_version
            ).encode("utf-8")
        ).hexdigest()[:16]

        library_path = GENERATED_DIR / (
            f"kernel_{mode}_{cache_key}{extension}"
        )

        cached_source_path = GENERATED_DIR / (
            f"kernel_{mode}_{cache_key}.cpp"
        )

        cached_source_path.write_text(cpp_code)

        print(f"\nGenerated: {source_path}")
        print(cpp_code)

        if library_path.exists():
            compilation_ms = 0.0
            print(f"CACHE HIT: {library_path.name}")
        else:
            compilation_ms = compile_cpp(
                cached_source_path, library_path, opt_level
            )
            print(f"CACHE MISS: compiled in {compilation_ms:.3f} ms")

        COMPILATION_RECORDS.append({
            "mode": mode,
            "compilation_ms": compilation_ms,
        })

        library = ctypes.CDLL(str(library_path))
        kernel = library.run_kernel

        kernel.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        kernel.restype = None

        def run(*args):
            inputs = [x for x in args if isinstance(x, torch.Tensor)]

            if len(inputs) != 1:
                raise NotImplementedError("Exactly one tensor input is supported.")

            x = inputs[0]

            if x.device.type != "cpu":
                raise NotImplementedError("Only CPU tensors are supported.")

            if x.dtype != torch.float32:
                raise NotImplementedError("Only float32 tensors are supported.")

            # Noncontiguous inputs are copied into contiguous storage.
            x = x.contiguous()

            output = torch.empty_like(x)

            input_ptr = ctypes.cast(
                x.data_ptr(), ctypes.POINTER(ctypes.c_float)
            )
            output_ptr = ctypes.cast(
                output.data_ptr(), ctypes.POINTER(ctypes.c_float)
            )

            kernel(input_ptr, output_ptr, x.numel())

            return (output,)

        return run

    return cpp_backend


# Preserve the default unfused backend.
cpp_backend = make_cpp_backend(fuse=False)
