import os, random
import bpy

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



    



if __name__ == "__main__":
    thingi10k_path = "/home/ola/library/Thingi10K/raw_meshes"
    print(random.choice(os.listdir(thingi10k_path)))



