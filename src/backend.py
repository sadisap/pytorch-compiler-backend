import ctypes
import platform
import subprocess
from pathlib import Path

import torch

from codegen import generate_cpp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = PROJECT_ROOT / "generated"


def compile_cpp(source_path, library_path):
    system = platform.system()

    if system == "Darwin":
        command = [
            "clang++",
            "-std=c++17",
            "-O0",
            "-dynamiclib",
            str(source_path),
            "-o",
            str(library_path),
        ]
    elif system == "Linux":
        command = [
            "g++",
            "-std=c++17",
            "-O0",
            "-shared",
            "-fPIC",
            str(source_path),
            "-o",
            str(library_path),
        ]
    else:
        raise RuntimeError(
            f"Unsupported operating system: {system}"
        )

    print("\n=== C++ Compile Command ===")
    print(" ".join(command))

    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def cpp_backend(gm, example_inputs):
    print("\n=== C++ Backend Called ===")

    print("\nFX Graph:")
    print(gm.graph)

    GENERATED_DIR.mkdir(exist_ok=True)

    source_path = GENERATED_DIR / "kernel.cpp"

    if platform.system() == "Darwin":
        library_path = GENERATED_DIR / "kernel.dylib"
    else:
        library_path = GENERATED_DIR / "kernel.so"

    cpp_code = generate_cpp(
        gm,
        source_path,
    )

    print("\n=== Generated C++ ===")
    print(cpp_code)

    compile_cpp(
        source_path,
        library_path,
    )

    print("\nC++ compilation successful.")
    print(f"Library: {library_path}")

    library = ctypes.CDLL(
        str(library_path)
    )

    kernel = library.run_kernel

    kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]

    kernel.restype = None

    def run(*args):
        tensor_inputs = [
            arg
            for arg in args
            if isinstance(arg, torch.Tensor)
        ]

        if len(tensor_inputs) != 1:
            raise RuntimeError(
                "The current backend expects one tensor input."
            )

        x = tensor_inputs[0]

        if x.device.type != "cpu":
            raise RuntimeError(
                "The current backend only supports CPU tensors."
            )

        if x.dtype != torch.float32:
            raise RuntimeError(
                "The current backend only supports float32 tensors."
            )

        x = x.contiguous()

        output = torch.empty_like(x)

        input_ptr = ctypes.cast(
            x.data_ptr(),
            ctypes.POINTER(ctypes.c_float),
        )

        output_ptr = ctypes.cast(
            output.data_ptr(),
            ctypes.POINTER(ctypes.c_float),
        )

        kernel(
            input_ptr,
            output_ptr,
            x.numel(),
        )

        return (output,)

    return run