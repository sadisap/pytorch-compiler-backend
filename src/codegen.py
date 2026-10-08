
import operator
from pathlib import Path

import torch


def generate_cpp(gm, output_path, fuse=False):
    nodes = list(gm.graph.nodes)

    placeholders = [n for n in nodes if n.op == "placeholder"]
    operations = [n for n in nodes if n.op == "call_function"]
    outputs = [n for n in nodes if n.op == "output"]

    if len(placeholders) != 1 or len(outputs) != 1:
        raise NotImplementedError("Only one input and one output are supported.")

    if not operations:
        raise NotImplementedError("At least one operation is required.")

    input_node = placeholders[0]
    output_node = outputs[0]

    if any(n.op not in ("placeholder", "call_function", "output") for n in nodes):
        raise NotImplementedError("Unsupported graph node.")

    # Only a single chain of elementwise operations is supported.
    current = input_node
    expressions = []

    for node in operations:
        if node.target in (operator.mul, operator.add):
            if len(node.args) != 2:
                raise NotImplementedError("Expected a tensor and a scalar.")

            source, scalar = node.args

            if source is not current or not isinstance(scalar, (int, float)):
                raise NotImplementedError(
                    "Only tensor-first scalar operations in a single chain are supported."
                )

            scalar = float(scalar)

            if not (float("-inf") < scalar < float("inf")):
                raise NotImplementedError("Non-finite scalar is unsupported.")

            if node.target == operator.mul:
                expressions.append(("mul", scalar))
            else:
                expressions.append(("add", scalar))

        elif node.target == torch.relu:
            if len(node.args) != 1 or node.args[0] is not current:
                raise NotImplementedError("Unsupported ReLU input.")

            expressions.append(("relu", None))

        else:
            raise NotImplementedError(f"Unsupported operation: {node.target}")

        current = node

    result = output_node.args[0]

    if not isinstance(result, (tuple, list)) or len(result) != 1:
        raise NotImplementedError("Expected one graph output.")

    if result[0] is not current:
        raise NotImplementedError("Unsupported graph output.")

    lines = [
        "#include <algorithm>",
        "#include <cstddef>",
        "#include <vector>",
        "",
        'extern "C" void run_kernel(',
        "    const float* input,",
        "    float* output,",
        "    std::size_t n",
        ") {",
    ]

    if fuse:
        expression = "input[i]"

        for op, scalar in expressions:
            if op == "mul":
                expression = f"({expression} * {scalar.hex()}f)"
            elif op == "add":
                expression = f"({expression} + {scalar.hex()}f)"
            else:
                expression = f"std::max({expression}, 0.0f)"

        lines.extend([
            "    for (std::size_t i = 0; i < n; ++i) {",
            f"        output[i] = {expression};",
            "    }",
        ])

    else:
        previous = "input"

        for index, (op, scalar) in enumerate(expressions):
            name = f"temp{index}"
            lines.append(f"    std::vector<float> {name}(n);")
            lines.append("    for (std::size_t i = 0; i < n; ++i) {")

            if op == "mul":
                expression = f"{previous}[i] * {scalar.hex()}f"
            elif op == "add":
                expression = f"{previous}[i] + {scalar.hex()}f"
            else:
                expression = f"std::max({previous}[i], 0.0f)"

            lines.append(f"        {name}[i] = {expression};")
            lines.append("    }")
            previous = name

        lines.extend([
            "    for (std::size_t i = 0; i < n; ++i) {",
            f"        output[i] = {previous}[i];",
            "    }",
        ])

    lines.extend(["}", ""])

    cpp_code = "\n".join(lines)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cpp_code)

    return cpp_code
