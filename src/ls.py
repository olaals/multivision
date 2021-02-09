import cv2
import numpy as np

def create_laser_scan_line(color, half_line_width, image_width, image_height):
    img = np.zeros((image_height, image_width, 3))
    img[:, int(image_width/2-half_line_width):int(image_width/2+half_line_width), 2] = 255
    return img




if __name__ == '__main__':
    img = create_laser_scan_line("red", 1, 300, 300)
    cv2.imshow("sfdf", img)
    cv2.waitKey(0)
