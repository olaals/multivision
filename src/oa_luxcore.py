import bpy
import oa_ls as oals
import oa_blender as oabl
import oa_bl_meshes as oams
import oa_robotics as oarb
import math
import numpy as np


def luxcore_setup():
    bpy.context.scene.render.engine = 'LUXCORE'


class LuxcoreProjector:
    def __init__(self, name, location=(0,0,0), orientation=(0,0,0), lumens=0, normalize_color_luminance=False, fov_rad=math.pi/6):
        self.spot = bpy.data.lights.new(name=name + "_spot", type='SPOT')
        self.light_object = bpy.data.objects.new(name=name +"_lightobj", object_data=self.spot)
        self.light_object.data.luxcore.light_unit = 'lumen'
        self.light_object.data.luxcore.lumen=lumens
        bpy.context.collection.objects.link(self.light_object)
        self.light_object.location = location
        self.light_object.rotation_euler = orientation
        self.light_object.data.luxcore.normalizebycolor=normalize_color_luminance

    def set_projector_image(self, image, numpy_image=True):
        if numpy_image:
            image = oabl.numpy_img_to_blender_img(image) # convert to blender image
        self.spot.luxcore.image = image

    def set_lumens(self, lumens):
        self.light_object.data.luxcore.lumen = lumens

    def get_lumens(self):
        return self.light_object.luxcore.lumen

    def set_location(self, location):
        self.light_object.location = location

    def get_location(self):
        return self.light_object.location

    def set_rotation_euler(self, rotation):
        self.light_object.rotation_euler = rotation

    def get_rotation_euler(self):
        return self.light_object.rotation_euler

    def set_parent(self, parent_obj):
        self.light_object.parent = parent_obj


class LuxcoreLaser(LuxcoreProjector):
    def __init__(self, name, location=(0,0,0), orientation=(0,0,0), lumens=0, fov_rad=math.pi/6, image_res_x=1000, image_res_y=1000, half_line_width_px=2, laser_color=(255,0,0)):
        super().__init__(name, location=location, orientation=orientation, lumens=lumens, normalize_color_luminance=True, fov_rad=fov_rad)
        laser_img = oals.create_laser_scan_line(laser_color, half_line_width_px, image_res_x, image_res_y)
        self.set_projector_image(laser_img)


        
    def set_laser_image(self, laser_color, half_line_width_px, image_res_x, image_res_y):
        laser_img = oals.create_laser_scan_line(laser_color, half_line_width_px, image_res_x, image_res_y)
        self.set_projector_image(laser_img)


class LuxcoreTemplateScanner:
    def __init__(self, name, lightsource, location=(0,0,0), orientation=(0,0,0), distance_cam_lightsource=0.2, angle=math.pi/20, cam_left=True):
        self.lightsource = lightsource
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

class LuxcoreLaserScanner(LuxcoreTemplateScanner):
    def __init__(self, name, location=(0,0,0), orientation=(0,0,0), distance_cam_laser=0.2, angle=math.pi/20, lumens=100, cam_left=True):
        super().__init__(name, LuxcoreLaser(name + "_laser", lumens=lumens), location=location, orientation=orientation, distance_cam_lightsource=distance_cam_laser, angle=angle, cam_left=cam_left)

class LuxcoreStructuredLightScanner(LuxcoreTemplateScanner):
    def __init__(self, name, location=(0,0,0), orientation=(0,0,0), distance_cam_laser=0.2, angle=math.pi/20, lumens=100, cam_left=True):
        super().__init__(name, LuxcoreProjector(name + "_proj", lumens=lumens), location=location, orientation=orientation, distance_cam_lightsource=distance_cam_laser, angle=angle, cam_left=cam_left)




    
    


    


