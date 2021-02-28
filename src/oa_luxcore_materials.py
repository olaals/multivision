import bpy
from oa_blender import numpy_img_to_blender_img

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

