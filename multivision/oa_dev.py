import bpy
import numpy as np

def print_all_nodes():
    for node_tree in bpy.data.node_groups:
        for node in node_tree.nodes:
            print(node)

def img_info(img):
    print("Type")
    print(type(img))
    try:
        print("Type 0]")
        print(type(img[0]))
        print("Type[0][0]")
        print(type(img[0][0]))
        print("Type[0][0][0]")
        print(type(img[0][0][0]))
    except:
        print("None")
    print("Image shape")
    print(np.shape(img))
    print("Max min")
    print(np.max(img), np.min(img))

