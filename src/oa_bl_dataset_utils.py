import os, random
import bpy
import cv2
import numpy as np

def import_random_stl(dir_path, dimensions=(1,1,1)):
    random_stl = random.choice(os.listdir(dir_path))
    full_path_stl = os.path.join(dir_path, random_stl)
    bpy.ops.import_mesh.stl(filepath=full_path_stl, global_scale=1.0)
    bl_obj_name = os.path.splitext(random_stl)[0]
    bl_obj = bpy.data.objects[bl_obj_name]
    bl_obj.scale[0]=1.0/bl_obj.dimensions[0]*dimensions[0]
    bl_obj.scale[1]=1.0/bl_obj.dimensions[1]*dimensions[1]
    bl_obj.scale[2]=1.0/bl_obj.dimensions[2]*dimensions[2]
    return bl_obj

def import_stl(path):
    #print(f'Path: {path}')
    bpy.ops.import_mesh.stl(filepath=path)
    bl_obj_name = os.path.basename(path)
    #print(f"Split: {bl_obj_name}")
    bl_obj_name = os.path.splitext(bl_obj_name)[0]
    bl_obj_name = bl_obj_name.capitalize()
    #print(f"Splitext: {bl_obj_name}")
    bl_obj = bpy.data.objects[bl_obj_name]
    return bl_obj

def row_wise_mean_index(img):
    try:
        img = np.average(img, axis=2)
    except:
        print("grayscale image")
    ret_img = np.zeros(np.shape(img))
    n_cols = np.shape(img)[1]
    col_inds = np.array(list(range(n_cols)))
    for r in range(len(img)):
        row = img[r]
        max_val = np.max(row)
        if max_val > 100:
            row_sum = np.sum(row)
            weighted_sum = np.dot(col_inds, row)/float(row_sum)
            ind = int(weighted_sum)
            ret_img[r,ind] = 255
    return ret_img

def convert_to_binary(img):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except:
        pass
    img[img>50] = 255
    img[img<=50] = 0
    return img














    



if __name__ == "__main__":
    img = cv2.imread("/home/ola/Pictures/img_screenshot_29.01.2021.png")
    img = row_wise_mean_index(img)
    cv2.imshow("fsdds", img)
    cv2.waitKey(0)



