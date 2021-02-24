import bpy
import os
import numpy as np
import builtins as __builtin__

import oa_sli as sli

def console_print(*args, **kwargs):
    for a in bpy.context.screen.areas:
        if a.type == 'CONSOLE':
            c = {}
            c['area'] = a
            c['space_data'] = a.spaces.active
            c['region'] = a.regions[-1]
            c['window'] = bpy.context.window
            c['screen'] = bpy.context.screen
            s = " ".join([str(arg) for arg in args])
            for line in s.split("\n"):
                bpy.ops.console.scrollback_append(c, text=line)

def print(*args, **kwargs):
    """Console print() function."""

    console_print(*args, **kwargs) # to py consoles
    __builtin__.print(*args, **kwargs) # to system console

def newFunc2():
    print("dobbel hei")

def getChildren(myObject):
    children = []
    for ob in bpy.data.objects:
        if ob.parent == myObject:
            children.append(ob)
    return children

def get_argument(arg):
    args = sys.argv
    for i in range(len(args)):
        if args[i] == arg:
            return args[i+1]
    print("Could not find argument for ", arg)
    return None

def render(filename, output_dir, res_x, res_y, use_denoising=False):
    bpy.context.scene.view_layers[0].cycles.use_denoising = use_denoising
    bpy.context.scene.render.image_settings.file_format='PNG'
    bpy.context.scene.render.filepath = os.path.join(output_dir, filename)
    bpy.context.scene.render.resolution_x = res_x
    bpy.context.scene.render.resolution_y = res_y
    bpy.ops.render.render(write_still=True)

def get_image_list_from_folder(directory):
    pattern_img_list = []
    file_name_dir_list = os.listdir(directory)
    file_name_dir_list.sort()
    for file_name in file_name_dir_list:
        image = bpy.data.images.load(os.path.join(directory, file_name))
        pattern_img_list.append(image)
    return pattern_img_list

def numpy_img_to_blender_img(numpy_img):
    size = np.shape(numpy_img)[1], np.shape(numpy_img)[0]
    image = bpy.data.images.new("blImg", width=size[0], height=size[1], alpha=False)
    pixels = [None] * size[0] * size[1]
    for x in range(size[0]):
        for y in range(size[1]):
            a = 1.0
            r = numpy_img[y,x,0]
            g = numpy_img[y,x,1]
            b = numpy_img[y,x,2]

            pixels[(y * size[0]) + x] = [r, g, b, a]

    pixels = [chan for px in pixels for chan in px]
    image.pixels = pixels
    return image

def get_camera_matrix():
    scene = bpy.context.scene
    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale # px
    height = scene.render.resolution_y * scale # px
    camdata = scene.camera.data

    focal = camdata.lens # mm
    sensor_width = camdata.sensor_width # mm
    sensor_height = camdata.sensor_height # mm
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camdata.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = width / sensor_width / pixel_aspect_ratio
        s_v = height / sensor_height
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        s_u = width / sensor_width
        s_v = height * pixel_aspect_ratio / sensor_height
    # parameters of intrinsic calibration matrix K
    alpha_u = focal * s_u
    alpha_v = focal * s_v
    u_0 = width / 2
    v_0 = height / 2
    skew = 0 # only use rectangular pixels
    K = np.array([
        [alpha_u,    skew, u_0],
        [      0, alpha_v, v_0],
        [      0,       0,   1]
    ])
    return K

class Projector:
    def __init__(self, position, rotation, name):
        bpy.ops.projector.create()
        self.proj = bpy.data.objects.get("Projector")
        self.proj.name = "Projector_"+name
        children = getChildren(self.proj)
        self.spotObj = children[0] 
        self.spotObj.name = "Projector_"+name + ".Spot"
        self.spot = bpy.data.lights.get("Spot")
        self.spot.name = "Spot_"+name
        self.proj.location = position
        self.proj.rotation_euler = rotation
        self.proj.proj_settings.throw_ratio = 1
        self.turn_off_projector()

    def __del__(self):
        print("deleting ptoj")

    def turn_off_projector(self):
        self.spot.energy = 0

    def turn_on_projector(self, power):
        self.spot.energy = power

    def apply_gray_code_pattern(self, pattern_number, width, height):
        gray_code = sli.create_gray_code_pattern(pattern_number, width, height)
        bl_img = numpy_img_to_blender_img(gray_code)
        self.proj.proj_settings.projected_texture = 'custom_texture'
        self.spot.node_tree.nodes.get("Image Texture").image = bl_img
        self.proj.proj_settings.use_custom_texture_res = True
        self.proj.proj_settings.use_custom_texture_res = False
        self.proj.proj_settings.use_custom_texture_res = True




   

        




