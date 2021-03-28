import numpy as np
import cv2
import screeninfo

def init_proj(window_name, screen_id):
    screen = screeninfo.get_monitors()[screen_id]
    width, height = screen.width, screen.height

    cv2.moveWindow(window_name, screen.x -1, screen.y-1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN)
    return width, height

