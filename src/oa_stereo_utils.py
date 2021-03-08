import numpy as np
import itertools

def draw_epipolar_lines(img_left, img_right):
    height = np.shape(img_left)[0]
    divisions = 40.0
    colors = [(255,0,0), (0,0,255), (0,255,0), (255,255,0), (255,255,255), (0,255,255)]
    color_generator = itertools.cycle(colors)
    step = int(np.floor(height/divisions))
    stop = int(divisions*step)
    img = np.hstack([img_left, img_right])

    for col in range(0,stop-1, step):
        img[col, :, :] = next(color_generator)

    
    return img


        


