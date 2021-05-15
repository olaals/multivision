from skimage.util import random_noise
import cv2
import numpy as np
from itertools import chain, repeat, cycle, islice
from scipy.interpolate import interp1d

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

def row_wise_mean_sum_where_nonzero(img):
    row_sums = np.sum(img,axis=1)
    row_sums_nonzero = row_sums[row_sums!=0]
    mean = np.mean(row_sums_nonzero)
    return row_sums,mean

def secdeg_momentum_subpix(img, mean_threshold=0.5):
    img_float = img.astype(np.float32)
    img_max = np.max(img_float, axis=1).astype(np.float32)
    img_max = np.where(img_max == 0, 1, img_max)
    row_sums,mean = row_wise_mean_sum_where_nonzero(img)
    norm_img = img_float / img_max[:,None]
    col_inds = np.indices(img.shape[:2])[1]
    I_2 = np.power(norm_img, 2)
    top = I_2*col_inds
    sum_top = np.sum(top, axis=1)
    sum_bot = np.sum(I_2, axis=1)
    xs_ic = np.divide(sum_top, sum_bot, out=np.zeros_like(sum_top, dtype=np.float32), where=row_sums>mean_threshold*mean)
    return xs_ic

def subpix_to_image(subpix_array, img_shape):
    subpix_array = subpix_array.copy()
    subpix_array = np.round(subpix_array)
    x_ind = subpix_array.astype(np.uint16)
    img = np.zeros(img_shape, dtype=np.uint8)
    inds = np.array(list(range(len(subpix_array))))
    img[inds, x_ind] = 255
    return img

def get_enlarged_subpix_comp(img, subpix_arr, factor=4):
    img_l = cv2.resize(img, (factor*img.shape[1], factor*img.shape[0]), interpolation=cv2.INTER_NEAREST)
    xs_l = np.linspace(0,img_l.shape[0]-1, img.shape[0], dtype=np.uint16)
    xs_l2 = np.linspace(0,img_l.shape[0]-1, img_l.shape[0], dtype=np.uint16)
    f = interp1d(xs_l, subpix_arr)
    subpix_l = f(xs_l2)*factor
    subpix_img_l = subpix_to_image(subpix_l, img_l.shape)
    stacked_l = np.dstack((img_l, subpix_img_l, np.zeros_like(subpix_img_l)))
    return stacked_l








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
