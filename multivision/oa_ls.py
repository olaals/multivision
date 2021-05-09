from skimage.util import random_noise
import cv2
import numpy as np
from itertools import chain, repeat, cycle, islice

def create_laser_scan_line(color, line_width, image_width, image_height):
    assert(len(color)==3)
    half_line_width_left = np.round(line_width/2)
    half_line_width_right = np.round(line_width/1.9)
    img = np.zeros((image_height, image_width, 3))
    img[:, int(image_width/2-half_line_width_left):int(image_width/2+half_line_width_right)] = color
    return img

def create_laser_scan_line_speckle(color, line_width, image_width, image_height, gaussian_kernel_width=None):
    if gaussian_kernel_width is None:
        gaussian_kernel_width = (line_width%2)+line_width+1
    

    laser_img = create_laser_scan_line(color, line_width, image_width, image_height)
    laser_img_blur = cv2.GaussianBlur(laser_img, (gaussian_kernel_width, gaussian_kernel_width),0)
    laser_img_blur /=255.0
    laser_img_speckle = random_noise(laser_img_blur, mode='speckle', seed=None, clip=True)
    return np.uint8(laser_img_speckle*255.0)

def create_laser_scan_line_periodical_color(colors_list,  step, image_width, image_height, line_width=1):
    #assert(line_width%2 == image_width%2)
    img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    len_colors = len(colors_list)
    numbers = list(range(len_colors))
    indeces_to_loop = list(chain.from_iterable(zip(*repeat(numbers, step))))
    color_index_list_to_loop = list(islice(cycle(indeces_to_loop), image_height))
    half_line_width_left = np.round(line_width/2)
    half_line_width_right = np.round(line_width/1.9)

    for r in range(image_height):
        img[r, int(image_width/2-half_line_width_left):int(image_width/2+half_line_width_right)] = colors_list[color_index_list_to_loop[r]]
    return img







if __name__ == '__main__':

    #img = create_laser_scan_line_speckle((0,0,255), 3, 200, 200)
    img = make_color_wheel_image(200, 200)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = filter_hsv(img, (100, 0, 0), (110, 255,255))


    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("sfdf", img)
    cv2.waitKey(0)
    cv2.imwrite("testimg.png", img)
    cv2.destroyAllWindows()
