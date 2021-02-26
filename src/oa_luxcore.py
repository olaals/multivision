import bpy
import oa_ls as oals
import oa_sli as oasli
import oa_blender as oabl
import oa_bl_meshes as oams
import oa_robotics as oarb
import math
import numpy as np
import os
import copy
import mathutils


def luxcore_setup(render_time=120):
    bpy.context.scene.render.engine = 'LUXCORE'
    bpy.context.scene.luxcore.halt.use_time = True
    bpy.context.scene.luxcore.halt.time = render_time
    bpy.context.scene.luxcore.config.path.depth_total = 2
    bpy.context.scene.luxcore.config.path.depth_glossy = 2
    bpy.context.scene.luxcore.config.path.depth_specular = 2
    bpy.context.scene.luxcore.config.path.depth_diffuse = 2
    bpy.context.scene.luxcore.config.device = 'OCL'
    bpy.context.scene.luxcore.config.path.hybridbackforward_enable = True
    bpy.context.scene.luxcore.denoiser.enabled = False
    bpy.context.scene.luxcore.viewport.denoise = False
    bpy.context.scene.luxcore.viewport.add_light_tracing = False


    
    

def assign_material(object, material):

    luxcore_mat_dict = {
        "Disney": "LuxCoreNodeMatDisney",
        "Mix": "LuxCoreNodeMatMix",
        "Matte": "LuxCoreNodeMatMatte",
        "Glossy": "LuxCoreNodeMatGlossy2",
        "Glass": "LuxCoreNodeMatGlass",
        "Null (Transparent)": "LuxCoreNodeMatNull",
        "Metal": "LuxCoreNodeMatMetal",
        "Mirror": "LuxCoreNodeMatMirror",
        "Glossy Translucent": "LuxCoreNodeMatGlossyTranslucent",
        "Matte Translucent": "LuxCoreNodeMatMatteTranslucent",
    }

    mat = bpy.data.materials.new(name=material)
    tree_name = "Nodes_" + mat.name
    node_tree = bpy.data.node_groups.new(name=tree_name, type="luxcore_material_nodes")
    mat.luxcore.node_tree = node_tree
    # User counting does not work reliably with Python PointerProperty.
    # Sometimes, the material this tree is linked to is not counted as user.
    #node_tree.use_fake_user = True

    nodes = node_tree.nodes
    output_node = nodes.new("LuxCoreNodeMatOutput")
    output_node.location = 300, 200
    output_node.select = False
    mat_node = nodes.new(luxcore_mat_dict[material])
    mat_node.location = 50, 200
    node_tree.links.new(mat_node.outputs[0], output_node.inputs[0])

    ###############################################################

    if object.material_slots:
        object.material_slots[obj.active_material_index].material = mat
    else:
        object.data.materials.append(mat)


def assign_mix_material(object, material1, material2, weight=0.5):

    luxcore_mat_dict = {
        "Disney": "LuxCoreNodeMatDisney",
        "Mix": "LuxCoreNodeMatMix",
        "Matte": "LuxCoreNodeMatMatte",
        "Glossy": "LuxCoreNodeMatGlossy2",
        "Glass": "LuxCoreNodeMatGlass",
        "Null (Transparent)": "LuxCoreNodeMatNull",
        "Metal": "LuxCoreNodeMatMetal",
        "Mirror": "LuxCoreNodeMatMirror",
        "Glossy Translucent": "LuxCoreNodeMatGlossyTranslucent",
        "Matte Translucent": "LuxCoreNodeMatMatteTranslucent",
    }

    mat = bpy.data.materials.new(name=material1+"_"+material2)
    tree_name = "Nodes_" + mat.name
    node_tree = bpy.data.node_groups.new(name=tree_name, type="luxcore_material_nodes")
    mat.luxcore.node_tree = node_tree
    # User counting does not work reliably with Python PointerProperty.
    # Sometimes, the material this tree is linked to is not counted as user.
    #node_tree.use_fake_user = True

    nodes = node_tree.nodes
    output_node = nodes.new("LuxCoreNodeMatOutput")
    output_node.location = 300, 200
    output_node.select = False
    mat_node1 = nodes.new(luxcore_mat_dict[material1])
    mat_node2 = nodes.new(luxcore_mat_dict[material2])
    mix_node = nodes.new(luxcore_mat_dict["Mix"])
    mat_node1.location = 20, 100
    mat_node2.location = 20, 300
    mix_node.location = 200, 200
    node_tree.links.new(mat_node1.outputs[0], mix_node.inputs[0])
    node_tree.links.new(mat_node2.outputs[0], mix_node.inputs[1])
    node_tree.links.new(mix_node.outputs[0], output_node.inputs[0])

    ###############################################################

    if object.material_slots:
        object.material_slots[obj.active_material_index].material = mat
    else:
        object.data.materials.append(mat)


class Axis:
    def __init__(self, parent):
        self.axis = bpy.data.objects.new( "empty", None )
        bpy.context.scene.collection.objects.link(self.axis)
        self.axis.empty_display_size = 1
        self.axis.empty_display_type = 'ARROWS'  
        self.axis.rotation_euler = (math.pi, 0, 0)
        self.axis.parent = parent




class ObjectTemplate:
    def __init__(self, object):
        self.object = object

    def set_location(self, location):
        self.object.location = location

    def get_location(self):
        return self.object.location

    def set_rotation(self, rotation, mode="euler"):
        self.object.rotation_euler = rotation

    def get_rotation(self, mode="euler"):
        return self.object.rotation_euler

    def set_parent(self, parent_obj):
        self.object.parent = parent_obj

class LuxcoreProjector(ObjectTemplate):
    def __init__(self, name, location=(0,0,0), orientation=(0,0,0), lumens=0, normalize_color_luminance=True, fov_rad=math.pi/6):
        self.name = name
        self.spot = bpy.data.lights.new(name=name + "_spot", type='SPOT')
        self.light_object = bpy.data.objects.new(name=name +"_lightobj", object_data=self.spot)
        super().__init__(self.light_object)
        self.light_object.data.luxcore.light_unit = 'lumen'
        self.light_object.data.luxcore.lumen=lumens
        bpy.context.collection.objects.link(self.light_object)
        self.light_object.location = location
        self.light_object.rotation_euler = orientation
        self.light_object.data.luxcore.normalizebycolor=normalize_color_luminance
        self.light_object.data.spot_size = fov_rad
        self.axis = Axis(self.light_object)

    def set_projector_image(self, image, numpy_image=True):
        if numpy_image:
            image = oabl.numpy_img_to_blender_img(image) # convert to blender image
        self.spot.luxcore.image = image
        self.projection_image = image

    def set_lumens(self, lumens):
        self.light_object.data.luxcore.lumen = lumens

    def get_lumens(self):
        return self.light_object.luxcore.lumen


    
    def save_projection_image_to_png(self, filename="projection_image.png"):
        image = self.projection_image
        image.filepath_raw = os.path.join(os.getcwd(), filename)
        image.file_format = 'PNG'
        image.save()



class LuxcoreLaser(LuxcoreProjector):
    def __init__(self, name, location=(0,0,0), orientation=(0,0,0), lumens=0, fov_rad=math.pi/6, image_res_x=1000, image_res_y=1000, half_line_width_px=1, laser_color=(255,0,0)):
        super().__init__(name, location=location, orientation=orientation, lumens=lumens, normalize_color_luminance=True, fov_rad=fov_rad)
        laser_img = oals.create_laser_scan_line(laser_color, half_line_width_px, image_res_x, image_res_y)
        self.set_projector_image(laser_img)

    def set_laser_image(self, laser_color, half_line_width_px, image_res_x, image_res_y):
        laser_img = oals.create_laser_scan_line(laser_color, half_line_width_px, image_res_x, image_res_y)
        self.set_projector_image(laser_img)
    
    def set_laser_image_periodical(self, colors_list, half_line_width, step, image_res_x, image_res_y):
        laser_img = oals.create_laser_scan_line_periodical_color(colors_list, half_line_width, step, image_res_x, image_res_y)
        self.set_projector_image(laser_img)


class Camera(ObjectTemplate):
    def __init__(self, name, location=(0,0,0), rotation=(0,0,0), resolution=(1920,1080)):
        self.resolution = resolution
        cam = bpy.data.cameras.new(name)
        self.camera = bpy.data.objects.new(name, cam)
        super().__init__(self.camera)
        bpy.context.collection.objects.link(self.camera)
        self.camera.location = location
        self.camera.rotation_euler = rotation
        self.axis = Axis(self.camera)

    def add_axis(self):
        self.axis = bpy.data.objects.new( "empty", None )
        self.axis.empty_display_size = 1
        self.axis.empty_display_type = 'ARROWS'  
        self.axis.rotation_euler = (math.pi, 0, 0)
        self.axis.parent = self.camera
        bpy.context.collection.objects.link(self.axis)

    def render_camera_image(self, filename, directory=None):
        scene = bpy.context.scene
        scene.camera = self.camera
        scene.render.resolution_x = self.resolution[0]
        scene.render.resolution_y = self.resolution[1]
        if directory is None:
            scene.render.filepath = os.path.join(os.getcwd(), filename)
        else:
            scene.render.filepath = os.path.join(os.getcwd(), directory, filename)
        bpy.ops.render.render(write_still=True)

    def get_camera_matrix(self):
        #scene = bpy.context.scene
        #scale = scene.render.resolution_percentage / 100
        width = self.resolution[0] #* scale # px
        height = self.resolution[1]# * scale # px
        camdata = self.camera.data
        focal = camdata.lens # mm
        sensor_width = camdata.sensor_width # mm
        sensor_height = camdata.sensor_height # mm
        #pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        pixel_aspect_ratio = 1.0
        #print("pixel aspect ratio")
        #print(pixel_aspect_ratio)
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
    
class LuxcoreTemplateScanner:
    def __init__(self, name, lightsource, location=(0,0,0), orientation=(0,0,0), distance_cam_lightsource=0.2, angle=math.pi/20, camera_resolution=(1920,1080), cam_left=True):
        self.lightsource = lightsource
        self.camera_resolution=camera_resolution
        self.camera = Camera(name+"_camera", resolution=camera_resolution)

        if cam_left:
            self.camera.set_location((-distance_cam_lightsource/2,0,0))
            self.lightsource.set_location((distance_cam_lightsource/2,0,0))
            self.camera.set_rotation((0, -angle/2,0))
            self.lightsource.set_rotation((0, angle/2,0))
        else:
            self.camera.set_location((distance_cam_lightsource/2,0,0))
            self.lightsource.set_location((-distance_cam_lightsource/2,0,0))
            self.camera.set_rotation((0,angle/2,0))
            self.lightsource.set_rotation((0, -angle/2,0))

        self.cube = oams.add_cuboid(name, (distance_cam_lightsource, distance_cam_lightsource/2, distance_cam_lightsource/2), (0,0,distance_cam_lightsource/4))
        self.lightsource.set_parent(self.cube)
        self.camera.set_parent(self.cube)
        self.cube.location = location
        self.cube.rotation_euler = orientation
    
    def get_essential_matrix(self):
        lightsource_quat = self.lightsource.get_rotation().to_quaternion()
        lightsource_loc = self.lightsource.get_location()
        cam_quat = self.camera.get_rotation().to_quaternion()
        cam_loc = self.camera.location
        R_W_L = lightsource_quat.to_matrix()
        R_L_W = R_W_L.transposed()
        R_L_C = lightsource_quat.rotation_difference(cam_quat).to_matrix()
        t_L_C_W = cam_loc - lightsource_loc
        t_L_C_L = R_L_W @t_L_C_W
        R_L_C = np.array(R_L_C)
        essential_matrix = R_L_C@oarb.vec_to_so3(t_L_C_L)
        return essential_matrix
    
    def get_rotation_cam_to_lightsource(self, mode="matrix"):
        cam_quat = self.camera.get_rotation().to_quaternion()
        lightsource_quat = self.lightsource.get_rotation().to_quaternion()
        quat_C_L = cam_quat.rotation_difference(lightsource_quat)
        if mode=="matrix":
            return quat_C_L.to_matrix()
        
        elif mode=="euler":
            return quat_C_L.to_euler('XYZ')
        
        elif mode=="quaternion":
            return quat_C_L
        
        else:
            raise Exception("get_rotation_cam_to_light_source: No mode for " + mode)
            return
    
    def get_translation_cam_to_lightsource(self):
        lightsource_loc_world = self.lightsource.get_location()
        cam_loc_world = self.camera.location
        R_C_W = self.camera.get_rotation().to_matrix().transposed()
        t_C_L_W = lightsource_loc_world - cam_loc_world
        t_C_L_C = R_C_W@t_C_L_W
        return t_C_L_C
"""
class LuxcoreTemplateScanner:
    def __init__(self, name, lightsource, location=(0,0,0), orientation=(0,0,0), distance_cam_lightsource=0.2, angle=math.pi/20, camera_resolution=(1920,1080), cam_left=True):
        self.lightsource = lightsource
        self.camera_resolution=camera_resolution
        cam = bpy.data.cameras.new(name+"_camera")
        self.camera = bpy.data.objects.new(name+"_camera", cam)
        bpy.context.collection.objects.link(self.camera)

        if cam_left:
            self.camera.location = (-distance_cam_lightsource/2,0,0)
            self.lightsource.set_location((distance_cam_lightsource/2,0,0))
            self.camera.rotation_euler = (0,-angle/2,0)
            self.lightsource.set_rotation_euler((0, angle/2,0))
        else:
            self.camera.location = (distance_cam_lightsource/2,0,0)
            self.lightsource.set_location((-distance_cam_lightsource/2,0,0))
            self.camera.rotation_euler = (0,angle/2,0)
            self.lightsource.set_rotation_euler((0, -angle/2,0))

        self.cube = oams.add_cuboid(name, (distance_cam_lightsource, distance_cam_lightsource/2, distance_cam_lightsource/2), (0,0,distance_cam_lightsource/4))
        self.lightsource.set_parent(self.cube)
        self.camera.parent = self.cube
        self.cube.location = location
        self.cube.rotation_euler = orientation
    
    def render_camera_image(self, filename):
        scene = bpy.context.scene
        scene.camera = self.camera
        scene.render.resolution_x = self.camera_resolution[0]
        scene.render.resolution_y = self.camera_resolution[1]
        scene.render.filepath = os.path.join(os.getcwd(), filename)
        bpy.ops.render.render(write_still=True)


    
    def get_camera_matrix(self):
        #scene = bpy.context.scene
        #scale = scene.render.resolution_percentage / 100
        width = self.camera_resolution[0] #* scale # px
        height = self.camera_resolution[1]# * scale # px
        camdata = self.camera.data
        focal = camdata.lens # mm
        sensor_width = camdata.sensor_width # mm
        sensor_height = camdata.sensor_height # mm
        #pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        pixel_aspect_ratio = 1.0
        #print("pixel aspect ratio")
        #print(pixel_aspect_ratio)
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
    
    
    def get_essential_matrix(self):
        lightsource_quat = self.lightsource.get_rotation_euler().to_quaternion()
        lightsource_loc = self.lightsource.get_location()
        cam_quat = self.camera.rotation_euler.to_quaternion()
        cam_loc = self.camera.location
        R_W_L = lightsource_quat.to_matrix()
        R_L_W = R_W_L.transposed()
        R_L_C = lightsource_quat.rotation_difference(cam_quat).to_matrix()
        t_L_C_W = cam_loc - lightsource_loc
        t_L_C_L = R_L_W @t_L_C_W
        R_L_C = np.array(R_L_C)
        essential_matrix = R_L_C@oarb.vec_to_so3(t_L_C_L)
        return essential_matrix
    
    def get_rotation_cam_to_lightsource(self, mode="matrix"):
        cam_quat = self.camera.rotation_euler.to_quaternion()
        lightsource_quat = self.lightsource.get_rotation_euler().to_quaternion()
        quat_C_L = cam_quat.rotation_difference(lightsource_quat)
        if mode=="matrix":
            return quat_C_L.to_matrix()
        
        elif mode=="euler":
            return quat_C_L.to_euler('XYZ')
        
        elif mode=="quaternion":
            return quat_C_L
        
        else:
            raise Exception("get_rotation_cam_to_light_source: No mode for " + mode)
            return
    
    def get_translation_cam_to_lightsource(self):
        lightsource_loc_world = self.lightsource.get_location()
        cam_loc_world = self.camera.location
        R_C_W = self.camera.rotation_euler.to_matrix().transposed()
        t_C_L_W = lightsource_loc_world - cam_loc_world
        t_C_L_C = R_C_W@t_C_L_W
        return t_C_L_C
"""

class LuxcoreLaserScanner(LuxcoreTemplateScanner):
    def __init__(self, name, location=(0,0,0), orientation=(0,0,0), distance_cam_laser=0.2, angle=math.pi/20, lumens=100, camera_resolution=(1920,1080),cam_left=True):
        super().__init__(name, LuxcoreLaser(name + "_laser", lumens=lumens), location=location, orientation=orientation, distance_cam_lightsource=distance_cam_laser, angle=angle, camera_resolution=camera_resolution, cam_left=cam_left)

class LuxcoreStructuredLightScanner(LuxcoreTemplateScanner):
    def __init__(self, name, location=(0,0,0), orientation=(0,0,0), distance_cam_laser=0.2, angle=math.pi/20, lumens=3500, camera_resolution=(1920,1080), laser_resolution=(1920,1080), cam_left=True):
        super().__init__(name, LuxcoreProjector(name + "_proj", lumens=lumens), location=location, orientation=orientation, distance_cam_lightsource=distance_cam_laser, angle=angle, camera_resolution=camera_resolution, cam_left=cam_left)
        blue_img = oasli.create_blue_img(laser_resolution[0], laser_resolution[1])
        self.lightsource.set_projector_image(blue_img)




    
    


    


