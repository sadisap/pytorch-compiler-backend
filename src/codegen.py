import operator
from pathlib import Path

import torch


def generate_cpp(gm, output_path):
    nodes = list(gm.graph.nodes)

    tensor_placeholders = [
        node
        for node in nodes
        if node.op == "placeholder"
    ]

    if len(tensor_placeholders) != 1:
        raise NotImplementedError(
            "The current backend supports exactly one tensor input."
        )

    input_node = tensor_placeholders[0]

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

    value_names = {
        input_node: "input"
    }

    temp_number = 0
    final_value = None

    for node in nodes:
        if node.op == "placeholder":
            continue

        if node.op == "call_function":
            temp_name = f"temp{temp_number}"
            temp_number += 1

            lines.append(
                f"    std::vector<float> {temp_name}(n);"
            )

            if node.target == operator.mul:
                source_node, scalar = node.args
                source_name = value_names[source_node]

                lines.extend([
                    "    for (std::size_t i = 0; i < n; ++i) {",
                    f"        {temp_name}[i] = "
                    f"{source_name}[i] * {float(scalar)}f;",
                    "    }",
                ])

            elif node.target == operator.add:
                source_node, scalar = node.args
                source_name = value_names[source_node]

                lines.extend([
                    "    for (std::size_t i = 0; i < n; ++i) {",
                    f"        {temp_name}[i] = "
                    f"{source_name}[i] + {float(scalar)}f;",
                    "    }",
                ])

            elif node.target == torch.relu:
                source_node = node.args[0]
                source_name = value_names[source_node]

                lines.extend([
                    "    for (std::size_t i = 0; i < n; ++i) {",
                    f"        {temp_name}[i] = "
                    f"std::max({source_name}[i], 0.0f);",
                    "    }",
                ])

            else:
                raise NotImplementedError(
                    f"Unsupported operation: {node.target}"
                )

            value_names[node] = temp_name
            final_value = temp_name

        elif node.op == "output":
            continue

        else:
            raise NotImplementedError(
                f"Unsupported FX node type: {node.op}"
            )

    if final_value is None:
        raise RuntimeError("No output value was generated.")

    lines.extend([
        "",
        "    for (std::size_t i = 0; i < n; ++i) {",
        f"        output[i] = {final_value}[i];",
        "    }",
        "}",
        "",
    ])

    cpp_code = "\n".join(lines)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cpp_code)

    return cpp_code