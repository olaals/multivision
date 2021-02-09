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

def turn_off_projector(proj_spot_name):
    spot = bpy.data.lights[proj_spot_name]
    spot.energy = 0

def turn_on_projector(proj_spot_name, power):
    spot = bpy.data.lights[proj_spot_name]
    spot.energy = power

