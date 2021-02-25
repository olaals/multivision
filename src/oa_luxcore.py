import bpy
import oa_ls as oals
import oa_blender as oabl
import oa_bl_meshes as oams
import math


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

class LuxcoreLaserScanner:
    def __init__(self, name, location=(0,0,0), orientation=(0,0,0), distance_cam_laser=0.2, angle=math.pi/20, lumens=100):
        self.laser = LuxcoreLaser(name + "_laser", lumens=lumens)
        cam = bpy.data.cameras.new(name+"_camera")
        self.camera = bpy.data.objects.new(name+"_camera", cam)
        bpy.context.collection.objects.link(self.camera)

        self.camera.location = (distance_cam_laser/2,0,0)
        self.laser.set_location((-distance_cam_laser/2,0,0))
        self.camera.rotation_euler = (0,angle/2,0)
        self.laser.set_rotation_euler((0, -angle/2,0))

        self.cube = oams.add_cuboid(name, (distance_cam_laser, distance_cam_laser/2, distance_cam_laser/2), (0,0,distance_cam_laser/4))
        self.laser.set_parent(self.cube)
        self.camera.parent = self.cube
