import cv2
from oa_robotics import *


def row_wise_max_mask(mat, row_sum_threshold=0):
    row_sums = np.sum(mat, axis=1, keepdims=True)>row_sum_threshold
    mask = np.where(row_sums, mat.max(axis=1,keepdims=1) == mat, 0)
    return mask


def row_wise_max_index_mask(mat):
    ind = np.indices(mat.shape)[1]
    img_filter = np.where(mat>0, ind, 0)
    print(img_filter)
    img_filter = row_wise_max_mask(img_filter)
    return img_filter

def shift_add_values(mat, shift_array):
    for shift in shift_array:
        mat += np.roll(mat, shift)
    return mat


