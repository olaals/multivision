import bpy

def print_all_nodes():
    for node_tree in bpy.data.node_groups:
        for node in node_tree.nodes:
            print(node)
