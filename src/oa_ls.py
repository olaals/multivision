import cv2
import numpy as np
from itertools import chain, repeat, cycle, islice

def create_laser_scan_line(color, half_line_width, image_width, image_height):
    assert(len(color)==3)
    img = np.zeros((image_height, image_width, 3))
    img[:, int(image_width/2-half_line_width):int(image_width/2+half_line_width), :] = color
    return img


def create_laser_scan_line_periodical_color(colors_list, half_line_width, step, image_width, image_height):
    img = np.zeros((image_height, image_width, 3))
    len_colors = len(colors_list)
    numbers = list(range(len_colors))
    indeces_to_loop = list(chain.from_iterable(zip(*repeat(numbers, step))))
    color_index_list_to_loop = list(islice(cycle(indeces_to_loop), image_height))
    

    for r in range(image_height):
        img[r, int(image_width/2-half_line_width):int(image_width/2+half_line_width)] = colors_list[color_index_list_to_loop[r]]
    return img





if __name__ == '__main__':
    img = create_laser_scan_line_periodical_color([(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)], 2, 20, 1000, 1000)
    cv2.imshow("sfdf", img)
    cv2.waitKey(0)
    cv2.imwrite("testimg.png", img)
    cv2.destroyAllWindows()
