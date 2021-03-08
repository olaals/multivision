import numpy as np

def numpy_img_to_blender_img2(numpy_img):
    size = np.shape(numpy_img)[1], np.shape(numpy_img)[0]
    #image = bpy.data.images.new("blImg", width=size[0], height=size[1], alpha=False)
    pixels = [None] * size[0] * size[1]
    for x in range(size[0]):
        for y in range(size[1]):
            a = 1000.0
            r = numpy_img[y,x,0]
            g = numpy_img[y,x,1]
            b = numpy_img[y,x,2]

            pixels[(y * size[0]) + x] = [r, g, b, a]

    pixels = [chan for px in pixels for chan in px]
    #image.pixels = pixels
    return pixels

def numpy_img_to_blender_img3(numpy_img):
    size = np.shape(numpy_img)[1], np.shape(numpy_img)[0]
    #image = bpy.data.images.new("blImg", width=size[0], height=size[1], alpha=False)
    pixels = np.dstack((numpy_img, np.ones(numpy_img.shape[:-1], dtype=np.uint8)*1000))
    pixels = pixels.flatten()
    pixels = list(pixels)


    #image.pixels = pixels
    return pixels

if __name__ == '__main__':
    np_img = np.ones((5,4,3))
    np_img_shape = np_img.shape
    height = np_img_shape[0]
    print(height)
    width = np_img_shape[1]
    print(width)
    channels = np_img_shape[2]
    print(channels)
    print(np_img_shape)
    iter1 = 0
    for row in range(height):
        for col in range(width):
            for chan in range(channels):
                np_img[row,col,chan]=iter1
                iter1 += 1
    print(np_img)

    bl_pix = numpy_img_to_blender_img3(np_img)
    print(np.shape(bl_pix))
    print(bl_pix)
    bl_pix = numpy_img_to_blender_img2(np_img)
    print(np.shape(bl_pix))
    print(bl_pix)




