import cv2
import numpy as np
from oa_bl_meshes import *
from oa_luxcore_materials import assign_texture_material
import matplotlib.pyplot as plt

def get_square_board_image(resolution, square_size, define_by_resolution=True):
    assert(resolution[0]%square_size ==0 and resolution[1]%square_size==0)
    hor_dim = int(resolution[0]/square_size)
    ver_dim = int(resolution[1]/square_size)
    cb = np.kron([[1, 0] * hor_dim, [0, 1] * hor_dim] * ver_dim, np.ones((square_size, square_size), dtype=np.uint8))
    cb = cb*255
    cb = np.stack((cb,)*3, axis=-1)
    cb = cb.astype(np.uint8)
    return cb

def get_square_board_image_sb(resolution, square_size):
    assert(square_size%2 == 0)
    half_square_size = int(square_size/2)
    cb = get_square_board_image(resolution,square_size)
    cb_size = np.shape(cb)
    begin = int(square_size/2)
    hor_end = int(cb_size[1] -square_size/2)
    ver_end = int(cb_size[0] - square_size/2)
    cb = cb[begin:ver_end,begin:hor_end, :]
    cb2 = np.ones(cb_size)*255
    cb2[begin:ver_end,begin:hor_end, :] = cb

    for row in range(3*half_square_size, hor_end+1, 2*square_size):
        cv2.circle(cb2, (row,int(square_size/2)), half_square_size, (0,0,0), -1)
        cv2.circle(cb2, (row-square_size,ver_end), half_square_size, (0,0,0), -1)

    for col in range(3*half_square_size, ver_end+1, 2*square_size):
        cv2.circle(cb2, (int(square_size/2), col), half_square_size, (0,0,0), -1)
        cv2.circle(cb2, (hor_end,col-square_size), half_square_size, (0,0,0), -1)

    return cb2

def spawn_calibration_board(name="Calibration_board", number_of_squares=(8,4), square_size_meter=0.1, location=(0,0,0), one_square_resolution=200):
    size = np.array(number_of_squares)*square_size_meter
    plane = add_plane(name, size, location=location)
    resolution = tuple(np.array(number_of_squares)*one_square_resolution)
    print("Resolution")
    print(resolution)
    print("One square size")
    print(one_square_resolution)
    checker_board = get_square_board_image(resolution, one_square_resolution)
    plt.imshow(checker_board)
    plt.show()
    assign_texture_material(plane, checker_board)











