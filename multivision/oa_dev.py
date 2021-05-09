import bpy
import numpy as np
import cv2


def cv2_imwrite(filename, img):
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img2)


def cv2_imread(filename, colorcode=1):
    img = cv2.imread(filename, colorcode)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img




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

