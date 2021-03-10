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
import cv2


def luxcore_setup(render_time=60):
    bpy.context.scene.render.engine = 'LUXCORE'
    bpy.context.scene.luxcore.halt.enable = True
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

def cycles_setup():
    bpy.context.scene.render.engine = 'CYCLES'
    


class Axis:
    def __init__(self, parent, rotation=math.pi):
        self.parent = parent
        self.axis = bpy.data.objects.new( "empty", None )
        bpy.context.collection.objects.link(self.axis)
        self.axis.empty_display_size = 1
        self.axis.empty_display_type = 'ARROWS'  
        self.axis.rotation_euler = (rotation, 0, 0)
        self.axis.parent = parent
    
    def get_rotation_parent(self):
        R_CA = self.axis.rotation_euler.copy()
        R_BC = self.parent.rotation_euler.copy()
        R_BA = R_BC.to_matrix()@R_CA.to_matrix()

        return R_BA.to_euler()
        

class ObjectTemplate:
    def __init__(self, object):
        self.__object = object

    def set_location(self, location):
        self.__object.location = location

    def get_location(self):
        return self.__object.location

    def set_rotation(self, rotation, mode="euler"):
        self.__object.rotation_euler = rotation

    def get_rotation(self, mode="euler"):
        return self.__object.rotation_euler

    def set_parent(self, parent_obj):
        self.__object.parent = parent_obj
    
    def get_parent(self):
        return self.__object.parent
    
    def look_at(self, look_at_point):
        location = self.__object.matrix_world.to_translation()
        look_at_point = mathutils.Matrix.Translation(look_at_point).to_translation()
        print(look_at_point)
        direction = look_at_point - location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        self.__object.rotation_euler = rot_quat.to_euler()

class CyclesProjector(ObjectTemplate):
    def __init__(self, name="CyclesProj", location=(0,0,0), orientation=(0,0,0), resolution = (1920,1080), focal_length=36, px_size_mm=10e-3, light_strength=1000):
        self.name = name
        self.focal_length = focal_length
        self.px_size_mm = px_size_mm
        self.resolution = resolution
        self.create_cycles_projector()
        super().__init__(self.light_object)
        bpy.context.collection.objects.link(self.light_object)
        self.light_object.location = location
        self.light_object.rotation_euler = orientation
        self.axis = Axis(self.light_object, rotation=math.pi)
        self.update_fov()
        self.set_default_image()
        self.projection_image = None

    def create_cycles_projector(self):
        spot = bpy.data.lights.new(self.name+"_spot", type="SPOT")
        self.spot = spot
        spot_obj = bpy.data.objects.new(self.name ,spot)
        #spot_obj.scale = (0.1,0.1,0.1) TODO: how to scale parent without scaling children (Axis becomes small)
        self.light_object = spot_obj
        spot.spot_size=3.14
        spot.energy = 1000
        spot.shadow_soft_size = 0
        #bpy.context.collection.objects.link(spot_obj)

        spot.use_nodes = True

        node_tree = spot.node_tree
        nodes = node_tree.nodes

        output_node = nodes["Light Output"]
        emission = nodes["Emission"]
        img_text_node = nodes.new("ShaderNodeTexImage")
        img_text_node.extension = 'CLIP'
        node_tree.links.new(img_text_node.outputs[0], emission.inputs[0])
        mapping_node = nodes.new("ShaderNodeMapping")
        mapping_node.inputs[1].default_value[0] = 0.5
        mapping_node.inputs[1].default_value[1] = 0.5
        mapping_node.inputs[2].default_value[2] = math.pi
        node_tree.links.new(mapping_node.outputs[0], img_text_node.inputs[0])
        vector_divide_node = nodes.new("ShaderNodeVectorMath")
        vector_divide_node.operation = 'DIVIDE'
        node_tree.links.new(vector_divide_node.outputs[0], mapping_node.inputs[0])
        seperate_XYZ_node = nodes.new("ShaderNodeSeparateXYZ")
        node_tree.links.new(seperate_XYZ_node.outputs[2], vector_divide_node.inputs[1])
        tex_coord_node = nodes.new("ShaderNodeTexCoord")
        node_tree.links.new(tex_coord_node.outputs[1], seperate_XYZ_node.inputs[0])
        node_tree.links.new(tex_coord_node.outputs[1], vector_divide_node.inputs[0])

        self.mapping_node = mapping_node
        self.img_text_node = img_text_node

    def set_projector_image(self, image, numpy_image=True):
        if numpy_image:
            self.projection_image = image
            image = oabl.numpy_img_to_blender_img(image) # convert to blender image

        self.img_text_node.image = image
    
    def update_fov(self):
        scale_x = self.focal_length/(self.resolution[0]*self.px_size_mm)
        scale_y = self.focal_length/(self.resolution[1]*self.px_size_mm)
        self.mapping_node.inputs[3].default_value[0] = scale_x
        self.mapping_node.inputs[3].default_value[1] = scale_y

    
    def get_image(self):
        pass

    def set_projector_parameters(self, focal_length, pix_size_mm, resolution):
        pass

    def get_camera_matrix(self):
        focal = self.focal_length
        u_0 = self.resolution[0]/2
        v_0 = self.resolution[1]/2
        K = np.array([
            [focal/self.pixel_size_mm,    0.0, u_0],
            [      0, focal/self.pixel_size_mm, v_0],
            [      0,       0,   1]
        ])
        return K

    def get_resolution(self):
        return self.resolution

    def set_default_image(self):
        proj_res = self.resolution
        blue_img = oasli.create_blue_img(proj_res[0], proj_res[1])
        self.set_projector_image(blue_img)


class LuxcoreProjector(ObjectTemplate):
    def __init__(self, name, location=(0,0,0), orientation=(0,0,0), lumens=1000, normalize_color_luminance=True, resolution = (1920,1080), focal_length=36, px_size_mm=10e-3):
        self.name = name
        self.focal_length = focal_length
        self.px_size_mm = px_size_mm
        self.resolution = resolution
        self.spot = bpy.data.lights.new(name=name + "_spot", type='SPOT')
        self.lumens = lumens
        self.light_object = bpy.data.objects.new(name=name +"_lightobj", object_data=self.spot)
        super().__init__(self.light_object)
        self.light_object.data.luxcore.light_unit = 'lumen'
        self.light_object.data.luxcore.lumen=lumens
        bpy.context.collection.objects.link(self.light_object)
        self.light_object.location = location
        self.light_object.rotation_euler = orientation
        self.light_object.data.luxcore.normalizebycolor=normalize_color_luminance
        self.axis = Axis(self.light_object)
        self.update_fov()
        self.projection_image = None

    def update_fov(self):
        theta = 2*np.arctan((self.px_size_mm*min(self.resolution[0], self.resolution[1]))/(2*self.focal_length))
        self.light_object.data.spot_size = theta

    def get_image(self):
        if self.projection_image is None:
            raise Exception("Projection image is None")
        return self.projection_image

    def set_projector_parameters(self, focal_length, px_size_mm, resolution):
        self.focal_length = focal_length
        self.px_size_mm = px_size_mm
        self.resolution = resolution
        self.update_fov()
    
    def turn_off_projector(self):
        self.light_object.luxcore.lumen = 0
    
    def turn_on_projector(self):
        self.light_object.luxcore.lumen = self.lumens



    def get_camera_matrix(self):
        focal = self.focal_length
        u_0 = self.resolution[0]/2
        v_0 = self.resolution[1]/2
        K = np.array([
            [focal/self.pixel_size_mm,    0.0, u_0],
            [      0, focal/self.pixel_size_mm, v_0],
            [      0,       0,   1]
        ])
        return K

    def set_projector_image(self, image, numpy_image=True):
        if numpy_image:
            self.projection_image = image
            image = oabl.numpy_img_to_blender_img(image) # convert to blender image
        else:
            self.projection_image = oabl.blender_img_to_numpy_img(image)
            
        self.spot.luxcore.image = image

    def set_lumens(self, lumens):
        self.lumens = lumens
        self.light_object.data.luxcore.lumen = lumens

    def get_lumens(self):
        return self.light_object.luxcore.lumen
    
    def save_projection_image_to_png(self, filename="projection_image.png"):
        image = self.projection_image
        image.filepath_raw = os.path.join(os.getcwd(), filename)
        image.file_format = 'PNG'
        image.save()
    
    def get_resolution(self):
        return self.resolution



class LuxcoreLaser(LuxcoreProjector):
    def __init__(self, name, location=(0,0,0), orientation=(0,0,0), lumens=0, resolution=(1920,1080), half_line_width_px=1, laser_color=(255,0,0), focal_length=36, px_size_mm=10e-3):
        super().__init__(name, location=location, orientation=orientation, lumens=lumens, normalize_color_luminance=True, resolution=resolution, focal_length=focal_length, px_size_mm=px_size_mm)
        laser_img = oals.create_laser_scan_line(laser_color, half_line_width_px, resolution[0], resolution[1])
        self.set_projector_image(laser_img)

    def set_laser_image(self, laser_color, half_line_width_px, image_res_x, image_res_y):
        laser_img = oals.create_laser_scan_line(laser_color, half_line_width_px, image_res_x, image_res_y)
        self.set_projector_image(laser_img)
    
    def set_laser_image_periodical(self, colors_list, half_line_width, step, image_res_x, image_res_y):
        laser_img = oals.create_laser_scan_line_periodical_color(colors_list, half_line_width, step, image_res_x, image_res_y)
        self.set_projector_image(laser_img)


class Camera(ObjectTemplate):
    def __init__(self, name, location=(0,0,0), rotation=(0,0,0), resolution=(1920,1080), focal_length=36, pixel_size_mm=10e-3):
        self.name = name
        self.resolution = resolution
        self.pixel_size_mm = pixel_size_mm
        cam = bpy.data.cameras.new(name)
        self.camera = bpy.data.objects.new(name, cam)
        super().__init__(self.camera)
        self.camera.data.lens = focal_length
        self.camera.data.sensor_fit = 'HORIZONTAL'
        self.camera.data.sensor_width = resolution[0]*pixel_size_mm
        self.camera.data.sensor_height = resolution[1]*pixel_size_mm
        bpy.context.collection.objects.link(self.camera)
        self.camera.location = location
        self.camera.rotation_euler = rotation
        self.axis = Axis(self.camera)

    def render(self, filename, directory=None):
        scene = bpy.context.scene
        scene.camera = self.camera
        scene.render.resolution_x = self.resolution[0]
        scene.render.resolution_y = self.resolution[1]
        if directory is None:
            scene.render.filepath = os.path.join(os.getcwd(), filename)
        else:
            scene.render.filepath = os.path.join(os.getcwd(), directory, filename)
        bpy.ops.render.render(write_still=True)
    
    def get_image(self):
        self.render("latest_render.png")
        img = cv2.imread("latest_render.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img



    def get_camera_matrix(self):
        #scene = bpy.context.scene
        #scale = scene.render.resolution_percentage / 100
        width = self.resolution[0] #* scale # px
        height = self.resolution[1]# * scale # px
        camdata = self.camera.data
        focal = camdata.lens # mm
        sensor_width = camdata.sensor_width # mm
        sensor_height = camdata.sensor_height # mm
        u_0 = width / 2
        v_0 = height / 2
        #skew = 0 # only use rectangular pixels
        K = np.array([
            [focal/self.pixel_size_mm,    0.0, u_0],
            [      0, focal/self.pixel_size_mm, v_0],
            [      0,       0,   1]
        ])
        return K


    
class StereoTemplate(ObjectTemplate):
    def __init__(self, name, left_optical, right_optical, location=(0,0,0), orientation=(0,0,0), intra_axial_dist=0.2, angle=math.pi/20):
        self.name = name
        self.cube = oams.add_cuboid(name, (intra_axial_dist, intra_axial_dist/2, intra_axial_dist/2), (0,0,intra_axial_dist/4))
        super().__init__(self.cube)
        self.__left_optical = left_optical
        self.__right_optical = right_optical
        self.__left_optical.set_location((-intra_axial_dist/2,0,0))
        self.__right_optical.set_location((intra_axial_dist/2,0,0))
        self.__left_optical.set_rotation((0, -angle/2,0))
        self.__right_optical.set_rotation((0, angle/2, 0))
        self.__left_optical.set_parent(self.cube)
        self.__right_optical.set_parent(self.cube)
        self.cube.location = location
        self.cube.rotation_euler = orientation
        
    def get_essential_matrix(self, right_to_left=True):
        transl_RL_R = self.get_translation_right_to_left_optical()
        rot_RL = self.get_rotation_right_to_left_optical()
        rot_RL = np.array(rot_RL)
        essential_matrix = rot_RL@oarb.vec_to_so3(transl_RL_R)
        return essential_matrix
    
    def get_rotation_left_to_right_optical(self, mode="matrix", return_numpy=False):
        rot_BL = self.__left_optical.axis.get_rotation_parent().to_matrix()
        rot_BR = self.__right_optical.axis.get_rotation_parent().to_matrix()
        rot_LB = rot_BL.transposed()
        rot_LR = rot_LB@rot_BR
        if mode=="matrix":
            if return_numpy:
                return np.array(rot_LR)
            else:
                return rot_LR
        
        elif mode=="euler":
            return rot_LR.to_euler('XYZ')
        
        elif mode=="quaternion":
            return rot_LR.to_quaternion()
        
        else:
            raise Exception("get_rotation_cam_to_light_source: No mode for " + mode)
            return

    def get_rotation_right_to_left_optical(self, mode="matrix", return_numpy=False):
        rot_BL = self.__left_optical.axis.get_rotation_parent().to_matrix()
        rot_BR = self.__right_optical.axis.get_rotation_parent().to_matrix()
        rot_RB = rot_BR.transposed()
        rot_RL = rot_RB@rot_BL
        if mode=="matrix":
            if return_numpy:
                return np.array(rot_RL)
            else:
                return rot_RL

        
        elif mode=="euler":
            return rot_RL.to_euler('XYZ')
        
        elif mode=="quaternion":
            return rot_RL.to_quaternion()
        
        else:
            raise Exception("get_rotation_cam_to_light_source: No mode for " + mode)
            return
    
    def get_translation_left_to_right_optical(self):
        transl_BR_B = self.__right_optical.get_location()
        transl_BL_B = self.__left_optical.get_location()
        rot_LB = self.__left_optical.axis.get_rotation_parent().to_matrix().transposed()
        transl_LR_B = transl_BR_B - transl_BL_B
        transl_LR_L = rot_LB@transl_LR_B
        return transl_LR_L

    def get_translation_right_to_left_optical(self, return_numpy=False):
        transl_BR_B = self.__right_optical.get_location()
        transl_BL_B = self.__left_optical.get_location()
        rot_BR = self.__right_optical.axis.get_rotation_parent().to_matrix()
        rot_RB = rot_BR.transposed()
        transl_RL_B = transl_BL_B - transl_BR_B
        transl_RL_R = rot_RB@transl_RL_B

        if return_numpy:
            return np.array(transl_RL_R)
        else:
            return transl_RL_R

    def get_rectified_image_pair(self, crop_parameter):
        left_img = self.__left_optical.get_image()
        right_img = self.__right_optical.get_image()
        left_img_size = left_img.shape[0:2][::-1]
        print("Left img size")
        print(left_img_size)
        print("type")
        print(type(left_img_size))
        right_img_size = right_img.shape[0:2][::-1]
        left_K = self.__left_optical.get_camera_matrix()
        right_K = self.__right_optical.get_camera_matrix()
        transl_RL_R = self.get_translation_right_to_left_optical(return_numpy=True)
        print("transl_RL_R")
        print(transl_RL_R)
        rot_RL = self.get_rotation_right_to_left_optical(return_numpy=True)
        print("rot_RL")
        print(rot_RL)
        distCoeffs = None

        R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(left_K, distCoeffs, right_K, distCoeffs, left_img_size, rot_RL, transl_RL_R, alpha=crop_parameter)
        left_maps = cv2.initUndistortRectifyMap(left_K, distCoeffs, R1, P1, left_img_size, cv2.CV_16SC2)
        right_maps = cv2.initUndistortRectifyMap(right_K, distCoeffs, R2, P2, right_img_size, cv2.CV_16SC2)
        left_img_remap = cv2.remap(left_img, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
        right_img_remap = cv2.remap(right_img, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)
        return left_img_remap, right_img_remap


    
class StereoCamera(StereoTemplate):
    def __init__(self, name, location=(0,0,0), orientation=(0,0,0), intra_axial_dist=0.2, angle=math.pi/20, focal_length=36, camera_resolution=(1920,1080)):
        self.left_cam = Camera(name + "_left_cam", focal_length=focal_length,resolution=camera_resolution)
        self.right_cam = Camera(name + "_right_cam", focal_length=focal_length, resolution=camera_resolution)
        super().__init__(name, self.left_cam, self.right_cam, location, orientation, intra_axial_dist, angle)
    
    def save_matrices_numpy(self, directory = "", print_matrices=False):
        left_cam_K = self.left_cam.get_camera_matrix()
        right_cam_K = self.right_cam.get_camera_matrix()
        transl_RL_R = self.get_translation_right_to_left_optical()
        transl_LR_L = self.get_translation_left_to_right_optical()
        essential = self.get_essential_matrix()
        rot_RL = self.get_rotation_right_to_left_optical()
        rot_LR = self.get_rotation_left_to_right_optical()

        np.save(os.path.join(directory, "left_cam_K"), left_cam_K)
        np.save(os.path.join(directory, "right_cam_K"), right_cam_K)
        np.save(os.path.join(directory, "transl_RL_R"), transl_RL_R)
        np.save(os.path.join(directory, "transl_LR_L"), transl_LR_L)
        np.save(os.path.join(directory, "rot_RL"), rot_RL)
        np.save(os.path.join(directory, "rot_LR"), rot_LR)
        np.save(os.path.join(directory, "essential"), essential)
        if print_matrices:
            print("Left camera matrix")
            print(left_cam_K)
            print("Right camera matrix")
            print(right_cam_K)
            print("Translation right to left camera")
            print(transl_RL_R)
            print("Translatiom left to right camera")
            print(transl_LR_L)
            print("Rotation left to right camera")
            print(rot_LR)
            print(rot_LR.to_euler())
            print("Rotation right to left camera")
            print(rot_RL)
            print(rot_RL.to_euler())
            print("Essential matrix")
            print(essential)




class LuxcoreLaserScanner(StereoTemplate):
    def __init__(self, name, location=(0,0,0), orientation=(0,0,0), intra_axial_dist=0.2, angle=math.pi/20, lumens=20, camera_resolution=(1920,1080),cam_left=True):
        self.camera = Camera(name + "_camera", resolution=camera_resolution)
        self.laser = LuxcoreLaser(name + "_laser", lumens=lumens)
        if cam_left:
            super().__init__(name, self.camera, self.laser, location, orientation, intra_axial_dist, angle)
        else:
            super().__init__(name, self.laser, self.camera, location, orientation, intra_axial_dist, angle)
    
    def get_filtered_scan(self):
        pass



class LuxcoreStructuredLightScanner(StereoTemplate):
    def __init__(self, name, location=(0,0,0), orientation=(0,0,0), intra_axial_dist=0.2, angle=math.pi/20, lumens=1000, cam_res=(1920,1080), proj_res=(1920, 1080), cam_left=True):
        self.camera = Camera(name + "_camera", resolution=cam_res)
        self.projector = LuxcoreLaser(name + "_laser", lumens=lumens)
        self.__cam_left = cam_left
        if cam_left:
            super().__init__(name, self.camera, self.projector, location, orientation, intra_axial_dist, angle)
        else:
            super().__init__(name, self.projector, self.camera, location, orientation, intra_axial_dist, angle)
        blue_img = oasli.create_blue_img(proj_res[0], proj_res[1])
        self.projector.set_projector_image(blue_img)

    
    def set_graycode_pattern(self, pattern_number=4):
        projector_res = self.projector.get_resolution()
        graycode_pattern = oasli.create_gray_code_pattern(pattern_number, projector_res[0], projector_res[1])
        self.projector.set_projector_image(graycode_pattern)
    
    def set_rainbow_pattern(self, pattern_number=4):
        projector_res = self.projector.get_resolution()
        rainbow_pattern = oasli.create_rainbow_pattern_img(pattern_number, projector_res[0], projector_res[1])
        self.projector.set_projector_image(rainbow_pattern)
        
        
    


