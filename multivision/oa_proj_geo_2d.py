import numpy as np
import cv2

def px_to_homg2d(px_coord, img_shape):
    px_coord = px_coord/px_coord[2]
    return np.array([px_coord[0], img_shape[1]-px_coord[1]-1, 1])

def homg2d_to_px(homg2d_coord, img_shape):
    homg2d_coord = homg2d_coord/homg2d_coord[2]
    px_coord = np.array([homg2d_coord[0], img_shape[1]-1-homg2d_coord[1], 1])
    return px_coord

def line2d_from_2_points(p1_homg2d, p2_homg2d):
    return np.cross(p1_homg2d, p2_homg2d)

def intersect_lines2d(l1_homg2d, l2_homg2d):
    p_homg2d = np.cross(l1_homg2d, l2_homg2d)
    p_homg2d = p_homg2d/p_homg2d[2]
    return p_homg2d

def draw_line2d(img, line_homg2d, color=(255,0,0), thickness=1):
    img_line = img

    vert1 = line2d_from_2_points(np.array([0,0,1]), np.array([0, 1, 1]))
    vert2 = line2d_from_2_points(np.array([img.shape[1], 0, 1]), np.array([img.shape[1], 1, 1]))

    p1 = intersect_lines2d(vert1, line_homg2d)
    p2 = intersect_lines2d(vert2, line_homg2d)

    px1 = homg2d_to_px(p1, img.shape).astype(np.uint16)
    px2 = homg2d_to_px(p2, img.shape).astype(np.uint16)
    img_line = cv2.line(img_line, (px1[0], px1[1]), (px2[0], px2[1]), color, thickness)

    return img_line



