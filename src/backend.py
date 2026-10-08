
import ctypes
import platform
import subprocess
import time
import uuid
from pathlib import Path

import torch

from codegen import generate_cpp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = PROJECT_ROOT / "generated"

COMPILATION_RECORDS = []


def compile_cpp(source_path, library_path):
    system = platform.system()

    if system == "Darwin":
        command = [
            "clang++", "-std=c++17", "-O0", "-dynamiclib",
            str(source_path), "-o", str(library_path),
        ]
    elif system == "Linux":
        command = [
            "g++", "-std=c++17", "-O0", "-shared", "-fPIC",
            str(source_path), "-o", str(library_path),
        ]
    else:
        raise RuntimeError(f"Unsupported operating system: {system}")

    start = time.perf_counter()
    subprocess.run(command, check=True, capture_output=True, text=True)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return elapsed_ms


def make_cpp_backend(fuse=False):
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

        source_path = GENERATED_DIR / f"kernel_{mode}.cpp"

        extension = ".dylib" if platform.system() == "Darwin" else ".so"

        unique_id = uuid.uuid4().hex

        library_path = GENERATED_DIR / f"kernel_{mode}_{unique_id}{extension}"

        cpp_code = generate_cpp(gm, source_path, fuse=fuse)

        print(f"\nGenerated: {source_path}")
        print(cpp_code)

        compilation_ms = compile_cpp(source_path, library_path)

        print(f"C++ compilation successful: {compilation_ms:.3f} ms")

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
