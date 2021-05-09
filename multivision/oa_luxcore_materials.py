import bpy
from oa_blender import numpy_img_to_blender_img
from oa_file_utils import *

def smart_project_uv(obj):
    lm = obj.data.uv_layers.new(name="LightMap") #new UV layer for lightmapping
    lm.active = True
    bpy.ops.object.editmode_toggle() 
    bpy.ops.mesh.select_all(action='SELECT') 
    bpy.ops.mesh.remove_doubles(threshold=0.001, use_unselected=False) 
    bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.222, user_area_weight=0.0, use_aspect=True, stretch_to_bounds=False)
    bpy.ops.object.editmode_toggle() 

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

    return mat_node

def assign_anisotropic(object, u_roughness, v_roughness):
    smart_project_uv(object)
    mat_node = assign_material(object, "Metal")
    mat_node.use_anisotropy = True
    mat_node.input_type = 'fresnel'
    mat_node.inputs[2].default_value = u_roughness
    mat_node.inputs[3].default_value = v_roughness

def assign_alu_low_matte(object, mix, u_roughness, v_roughness):
    smart_project_uv(object)
    matnode_alu, matnode_matte = assign_mix_material(object, "Metal", "Matte", mix)
    matnode_alu.use_anisotropy = True
    matnode_alu.input_type = 'fresnel'
    matnode_alu.inputs[2].default_value = u_roughness
    matnode_alu.inputs[3].default_value = v_roughness




def assign_pbr_material(object, pbr_dir_path):
    material_name = os.path.basename(pbr_dir_path)
    
    color_img_path = search_substring_from_folder(pbr_dir_path, "Color")
    metalness_img_path = search_substring_from_folder(pbr_dir_path, "Metalness")
    normal_img_path = search_substring_from_folder(pbr_dir_path, "Normal")
    roughness_img_path = search_substring_from_folder(pbr_dir_path, "Roughness")
    
    mat = bpy.data.materials.new(name=material_name)
    tree_name = "Nodes_" + mat.name
    node_tree = bpy.data.node_groups.new(name=tree_name, type="luxcore_material_nodes")
    mat.luxcore.node_tree = node_tree
    node_tree.use_fake_user = True

    nodes = node_tree.nodes
    output_node = nodes.new("LuxCoreNodeMatOutput")
    output_node.location = 500, 200
    output_node.select = False
    mat_node = nodes.new("LuxCoreNodeMatDisney")
    mat_node.location = 250, 200
    node_tree.links.new(mat_node.outputs[0], output_node.inputs[0])
    
    color_img_text_node = nodes.new("LuxCoreNodeTexImagemap")
    color_img_text_node.location = 0,700
    color_img_text_node.image = bpy.data.images.load(color_img_path)
    color_img_text_node.projection = 'box'
    node_tree.links.new(color_img_text_node.outputs[0], mat_node.inputs[0])
    
    metalness_img_text_node = nodes.new("LuxCoreNodeTexImagemap")
    metalness_img_text_node.location = -300, 400
    metalness_img_text_node.image = bpy.data.images.load(metalness_img_path)
    metalness_img_text_node.projection = 'box'
    metalness_img_text_node.gamma = 1
    node_tree.links.new(metalness_img_text_node.outputs[0], mat_node.inputs[2])
    
    roughness_img_text_node = nodes.new("LuxCoreNodeTexImagemap")
    roughness_img_text_node.location = 0, 100
    roughness_img_text_node.image = bpy.data.images.load(roughness_img_path)
    roughness_img_text_node.projection = 'box'
    roughness_img_text_node.gamma = 1
    node_tree.links.new(roughness_img_text_node.outputs[0], mat_node.inputs[5])
    
    normal_img_text_node = nodes.new("LuxCoreNodeTexImagemap")
    normal_img_text_node.location = -300, -300
    normal_img_text_node.image = bpy.data.images.load(normal_img_path)
    normal_img_text_node.projection = 'box'
    normal_img_text_node.gamma = 1
    node_tree.links.new(normal_img_text_node.outputs[0], mat_node.inputs[15])
    normal_img_text_node.is_normal_map = True
    

    ###############################################################

    if object.material_slots:
        object.material_slots[obj.active_material_index].material = mat
    else:
        object.data.materials.append(mat)


def assign_texture_material(object, image, mat_name="Image Texture", numpy_image=True):
    if numpy_image:
        image = numpy_img_to_blender_img(image)
    
    mat = bpy.data.materials.new(name=mat_name)
    tree_name = "Nodes_" + mat.name
    node_tree = bpy.data.node_groups.new(name=tree_name, type="luxcore_material_nodes")
    mat.luxcore.node_tree = node_tree
    node_tree.use_fake_user = True

    nodes = node_tree.nodes
    output = nodes.new("LuxCoreNodeMatOutput")
    output.location = 300, 200
    output.select = False

    glossy2 = nodes.new("LuxCoreNodeMatGlossy2")
    glossy2.location = 50, 200



    node_tree.links.new(glossy2.outputs[0], output.inputs[0])

    # Create imagemap node
    diffuse_img_node = nodes.new("LuxCoreNodeTexImagemap")
    diffuse_img_node.location = -200, 200
    diffuse_img_node.image = image
    node_tree.links.new(diffuse_img_node.outputs[0], glossy2.inputs["Diffuse Color"])

    map2d = nodes.new("LuxCoreNodeTexMapping2D")
    map2d.location = -300, 200
    #bpy.data.mesh.uv_layers.new(name='NewUVMap')
    print(dir(map2d))
    print(dir(map2d.uvmap))
    #bpy.ops.mesh.uv_texture_add()
    #node_tree.links.new(map2d.outputs[0], diffuse_img_node.inputs[0])

    # Assign to object (if you want)
    if object.material_slots:
        object.material_slots[object.active_material_index].material = mat
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
    mix_node.inputs[2].default_value = weight
    mat_node1.location = 20, 100
    mat_node2.location = 20, 300
    mix_node.location = 200, 200
    node_tree.links.new(mat_node1.outputs[0], mix_node.inputs[0])
    node_tree.links.new(mat_node2.outputs[0], mix_node.inputs[1])
    node_tree.links.new(mix_node.outputs[0], output_node.inputs[0])

    ###############################################################

    if object.material_slots:
        object.material_slots[object.active_material_index].material = mat
    else:
        object.data.materials.append(mat)

    return mat_node1, mat_node2


