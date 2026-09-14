def inspect_backend(gm, example_inputs):
    print("\n=== Custom backend called ===")

    print("\nFX Graph:")
    print(gm.graph)

    print("\nGraph Nodes:")
    for node in gm.graph.nodes:
        print(
            f"name={node.name}, "
            f"op={node.op}, "
            f"target={node.target}"
        )

    print("\n=== End backend inspection ===\n")

    return gm.forward