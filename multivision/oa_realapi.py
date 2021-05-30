import numpy as np
import cv2
import screeninfo
import oa_ls

def init_proj(window_name, screen_id):
    screen = screeninfo.get_monitors()[screen_id]
    width, height = screen.width, screen.height

    cv2.moveWindow(window_name, screen.x -1, screen.y-1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN)
    return width, height

def show_laser_line():
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    width, height = init_proj("window", 1)
    img_ls = create_laser_scan_line_speckle((0,0,255), 1, width, height, 3)
    cv2.imshow("window", img_ls)
    cv2.waitKey(0)
 

if __name__ == '__main__':
    show_laser_line()




