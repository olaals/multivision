import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import colorsys
from scipy.signal import convolve2d

def nothing(x):
    pass
    #print(x)

def d3stack(mask):
    return np.dstack((mask,mask,mask))

def filter_value(img, value):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = img_hsv[:,:,2]>value
    return np.where(d3stack(mask), img, 0)

def filter_similar_hue_multicolor(img1, img2, colors, hue_threshold, min_saturation=10, min_value=10,  pad=1):
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
    img1_hue = img1_hsv[:,:,0]
    img2_hue = img2_hsv[:,:,0]

    mask = np.zeros_like(img1_hue, dtype=np.bool)
    mask2 = img1_hsv[:,:,2]>0
    convolve_mat = np.ones((pad*2+1, pad*2+1), dtype=np.bool)

    for color in colors:
        color = np.array(color)
        norm_col = color/255
        hsv_norm = colorsys.rgb_to_hsv(norm_col[0], norm_col[1], norm_col[2])
        hsv_norm = np.array(hsv_norm)
        hsv = (hsv_norm*180).astype(np.int16)
        hue_low = hsv[0] - hue_threshold
        hue_high = hsv[0] + hue_threshold
        if hue_low<0:
            hue_low = 180+hue_low
        if hue_high>180:
            hue_high = hue_high - 180
        masked_img1 = filter_hsv(img1, (int(hue_low), min_saturation, min_value), (int(hue_high), 255,255), return_mask_only=True)
        masked_img2 = filter_hsv(img2, (int(hue_low), min_saturation, min_value), (int(hue_high), 255,255), return_mask_only=True)
        filtered = np.bitwise_and(masked_img1, masked_img2)

        filtered_pad = convolve2d(filtered, convolve_mat, 'same')
        filtered_pad = np.where(filtered_pad>0, True, False)
        mask = np.bitwise_and(filtered_pad | mask, mask2)

    return mask



def filter_hsv(image, lower_hsv, upper_hsv, to_grayscale=True, return_mask_only=False):
    result = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if lower_hsv[0] == upper_hsv[0]:
        return image

    if lower_hsv[0] >= upper_hsv[0]:
        lower_hsv1 = np.array([lower_hsv[0], lower_hsv[1], lower_hsv[2]])
        upper_hsv1 = np.array([180,upper_hsv[1], upper_hsv[2]])
        lower_hsv2 = np.array([0, lower_hsv[1], lower_hsv[2]])
        upper_hsv2 = np.array([upper_hsv[0], upper_hsv[1], upper_hsv[2]])


        mask1 = cv2.inRange(image, lower_hsv1, upper_hsv1)
        mask2 = cv2.inRange(image, lower_hsv2, upper_hsv2)
        mask = mask1 | mask2
    else:
        mask = cv2.inRange(image, lower_hsv, upper_hsv)

    result = cv2.bitwise_and(result, result, mask=mask)
    if to_grayscale:
        result = np.max(result, axis=2)



    if return_mask_only:
        return mask
    else:
        return result

def row_wise_max_mask(mat, row_sum_threshold=0):
    row_sums = np.sum(mat, axis=1, keepdims=True)>row_sum_threshold
    mask = np.where(row_sums, mat.max(axis=1,keepdims=1) == mat, 0)
    mask = mask.astype(np.bool)
    return mask


def row_wise_max_index_mask(mat):
    ind = np.indices(mat.shape)[1]
    img_filter = np.where(mat>0, ind, 0)
    img_filter = row_wise_max_mask(img_filter)
    return img_filter

def shift_add_values(mat, shift_array):
    for shift in shift_array:
        mat += np.roll(mat, shift)
    return mat

def shift_add_horizontal(mat, shift_num, right_to_left=True):
    result = mat
    print("shift_num", shift_num)
    for x_shift in range(shift_num-1):
        copy_mat = mat.copy()
        print("average line with result")
        print(get_average_line_width(result))
        print("average line with mat")
        print(get_average_line_width(mat))

        result += np.bitwise_or(mat, np.roll(copy_mat, (0, -1)))
    result = result.astype(np.bool)
    return result


def get_average_line_width(img, ceil=True):
    masked = np.where(img>0, 1, 0)
    row_wise_sum = np.sum(masked, axis=1)
    average_line_width = np.mean(row_wise_sum)
    if ceil:
        result = np.ceil(average_line_width)
    else:
        result = np.floor(average_line_width)
    return int(result)





def make_color_wheel_image(img_width, img_height):
    # source: https://stackoverflow.com/questions/65609247/create-color-wheel-pattern-image-in-python
    """
    Creates a color wheel based image of given width and height
    Args:
        img_width (int):
        img_height (int):

    Returns:
        opencv image (numpy array): color wheel based image
    """
    hue = np.fromfunction(lambda i, j: (np.arctan2(i-img_height/2, img_width/2-j) + np.pi)*(180/np.pi)/2,
                          (img_height, img_width), dtype=np.float)
    saturation = np.ones((img_height, img_width)) * 255
    value = np.ones((img_height, img_width)) * 255
    hsl = np.dstack((hue, saturation, value))
    color_map = cv2.cvtColor(np.array(hsl, dtype=np.uint8), cv2.COLOR_HSV2BGR)
    return color_map

def HSV_trackbar(value_name, trackbar_name, start_val):
    cv2.createTrackbar(f'{value_name} HUE', trackbar_name, min(start_val,180), 180, nothing)
    cv2.createTrackbar(f'{value_name} SAT', trackbar_name, start_val, 255, nothing)
    cv2.createTrackbar(f'{value_name} VAL', trackbar_name, start_val, 255, nothing)

def init_HSV_trackbar(value_name, trackbar_name, start_vals):
    cv2.createTrackbar(f'{value_name} HUE', trackbar_name, start_vals[0], 180, nothing)
    cv2.createTrackbar(f'{value_name} SAT', trackbar_name, start_vals[1], 255, nothing)
    cv2.createTrackbar(f'{value_name} VAL', trackbar_name, start_vals[2], 255, nothing)

def read_HSV_trackbar(value_name, trackbar_name):
    h = cv2.getTrackbarPos(f'{value_name} HUE', trackbar_name)
    s = cv2.getTrackbarPos(f'{value_name} SAT', trackbar_name)
    v = cv2.getTrackbarPos(f'{value_name} VAL', trackbar_name)
    return np.array([h,s,v])

def sum_channels_if_bitwise_nonzero(first_img, second_img):
    indices = np.bitwise_and(first_img>0, second_img>0)
    new_img = np.zeros_like(first_img)
    new_img[indices] = first_img[indices] + second_img[indices]
    new_img = np.clip(new_img, 0, 255)
    return nnew_img


def average_channels_if_bitwise_nonzero(first_img, second_img):
    indices = np.bitwise_and(first_img>0, second_img>0)
    ind_img = np.zeros_like(first_img, dtype=np.bool)
    ind_img[indices] = True
    stacked = np.dstack((first_img, second_img))
    averaged = np.average(stacked, axis=2)
    new_img = np.where(ind_img, averaged, 0)
    new_img = np.clip(new_img, 0, 255)
    new_img = new_img.astype(np.uint8)
    return new_img

def get_bitwise_nonzero_mask(first_img, second_img):
    first_img_mask = first_img>0
    second_img_mask = second_img>0
    result = np.zeros_like(first_img_mask, dtype=np.bool)
    inds = np.bitwise_and(first_img_mask, second_img_mask)
    result[inds] = True
    return result


    

def filter_image_with_trackbar(img, low_HSV_start=(0,0,0), high_HSV_start=(180,255,255)):
    cv2.namedWindow('image')
    cv2.namedWindow('colorwheel')
    init_HSV_trackbar('LOW1', 'colorwheel', low_HSV_start)
    init_HSV_trackbar('HIGH1', 'colorwheel', high_HSV_start)
    switch = '0 : NO SAVE\n 1 : SAVE'
    cv2.createTrackbar(switch, 'colorwheel', 0, 1, nothing)
    
    while(1):

        lower_hsv_1 = read_HSV_trackbar('LOW1', 'colorwheel')
        higher_hsv_1 = read_HSV_trackbar('HIGH1', 'colorwheel')
        img_out = filter_hsv(img, lower_hsv_1, higher_hsv_1, False)
        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", img_out)


        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            return lower_hsv_1, higher_hsv_1

    cv2.destroyAllWindows()

def filter_with_trackbar(img):
    time.sleep(1)
    #img = cv2.imread(image_path)
    #img = cv2.resize(img, (960, 540))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_colorwheel = make_color_wheel_image(200, 200)
    img_colorwheel_hsv = cv2.cvtColor(img_colorwheel, cv2.COLOR_BGR2HSV)

    cv2.namedWindow('image')
    cv2.namedWindow('colorwheel')
    HSV_trackbar('LOW1', 'colorwheel', 0)
    HSV_trackbar('HIGH1', 'colorwheel', 255)
    HSV_trackbar('LOW2', 'colorwheel', 255)
    HSV_trackbar('HIGH2', 'colorwheel', 255)
    HSV_trackbar('LOW3', 'colorwheel', 255)
    HSV_trackbar('HIGH3', 'colorwheel', 255)
    switch = '0 : NO SAVE\n 1 : SAVE'
    cv2.createTrackbar(switch, 'colorwheel', 0, 1, nothing)
    
    while(1):
        save = cv2.getTrackbarPos(switch, 'colorwheel')

        lower_hsv_1 = read_HSV_trackbar('LOW1', 'colorwheel')
        higher_hsv_1 = read_HSV_trackbar('HIGH1', 'colorwheel')
        mask1 = cv2.inRange(img_hsv, lower_hsv_1, higher_hsv_1)
        lower_hsv_2 = read_HSV_trackbar('LOW2', 'colorwheel')
        higher_hsv_2 = read_HSV_trackbar('HIGH2', 'colorwheel')
        mask2 = cv2.inRange(img_hsv, lower_hsv_2, higher_hsv_2)
        lower_hsv_3 = read_HSV_trackbar('LOW3', 'colorwheel')
        higher_hsv_3 = read_HSV_trackbar('HIGH3', 'colorwheel')
        mask3 = cv2.inRange(img_hsv, lower_hsv_3, higher_hsv_3)
        mask = mask1 | mask2 | mask3
        bitwise_and = cv2.bitwise_and(img, img, mask=mask)
        mask_cw1 = cv2.inRange(img_colorwheel_hsv, lower_hsv_1, higher_hsv_1)
        mask_cw2 = cv2.inRange(img_colorwheel_hsv, lower_hsv_2, higher_hsv_2)
        mask_cw3 = cv2.inRange(img_colorwheel_hsv, lower_hsv_3, higher_hsv_3)
        mask_cw = mask_cw1 | mask_cw2 | mask_cw3
        img_colorwheel_masked = cv2.bitwise_and(img_colorwheel, img_colorwheel, mask=mask_cw)
        resized = cv2.resize(bitwise_and, (960, 540))
        binary = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _,binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY)
        binary = cv2.resize(binary, (480, 270))

        cv2.imshow("binary", binary)
        cv2.imshow("image", resized)
        cv2.imshow("colorwheel", img_colorwheel_masked)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            if save:
                blue_img = np.zeros((1080,1920,3), np.uint8)
                red_img = np.zeros((1080,1920,3), np.uint8)
                green_img = np.zeros((1080,1920,3), np.uint8)
                blue_img[:] = (255,0,0)
                green_img[:] = (0,255,0)
                red_img[:] = (0,0,255)
                blue_part = cv2.bitwise_and(blue_img, blue_img, mask=mask1)
                green_part = cv2.bitwise_and(green_img, green_img, mask=mask2)
                red_part = cv2.bitwise_and(red_img, red_img, mask=mask3)
                result = blue_part+green_part+red_part

                cv2.destroyAllWindows()
                return result
            break

        

    cv2.destroyAllWindows()

def main():
    #sample_code()
    img = cv2.imread("colorwheel_hsv.jpg")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    low, up = filter_image_with_trackbar(img)
    print("low", low)
    print("up", up)
    img = filter_hsv(img, low, up)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("filtered.png", img)
    

def test_avg_channels():
    img1 = cv2.imread("testimg.png", 0)
    img1 = cv2.resize(img1, (10, 10))
    img2 = img1*1.5
    img2[0,9] = 255
    cv2.imshow("", img1)
    cv2.waitKey(0)
    cv2.imshow("", img2)
    cv2.waitKey(0)
    img3 = average_channels_if_bitwise_nonzero(img1, img2)
    cv2.imshow("", img3)
    cv2.waitKey(0)

    
def test_get_avg_line_width():
    img1 = cv2.imread("testimg.png", 0)
    img1 = cv2.resize(img1, (100, 100))
    cv2.imshow("", img1)
    cv2.waitKey(0)
    avg_line_width = get_average_line_width(img1)
    print(avg_line_width)





if __name__ == '__main__':
    #test_avg_channels()
    test_get_avg_line_width()

