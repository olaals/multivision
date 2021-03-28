import numpy as np
import itertools
import cv2

def draw_epipolar_lines(img_left, img_right):
    height = np.shape(img_left)[0]
    divisions = 40.0
    colors = [(255,0,0), (0,0,255), (0,255,0), (255,255,0), (255,255,255), (0,255,255)]
    color_generator = itertools.cycle(colors)
    step = int(np.floor(height/divisions))
    stop = int(divisions*step)
    img = np.hstack([img_left, img_right])

    for col in range(0,stop-1, step):
        img[col, :, :] = next(color_generator)

    
    return img

def rectify_images(left_img, right_img, left_K, right_K, transl_RL_R, rot_RL, crop_parameter):
    left_img_size = left_img.shape[0:2][::-1]
    right_img_size = right_img.shape[0:2][::-1]
    distCoeffs = None

    R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(left_K, distCoeffs, right_K, distCoeffs, left_img_size, rot_RL, transl_RL_R, alpha=crop_parameter)
    left_maps = cv2.initUndistortRectifyMap(left_K, distCoeffs, R1, P1, left_img_size, cv2.CV_16SC2)
    right_maps = cv2.initUndistortRectifyMap(right_K, distCoeffs, R2, P2, right_img_size, cv2.CV_16SC2)
    left_img_remap = cv2.remap(left_img, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
    right_img_remap = cv2.remap(right_img, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)
    return left_img_remap, right_img_remap



def filter_images(bright_img, no_light_img, treshold=0):
    mask = (bright_img<(no_light_img+treshold))
    filtered_img = bright_img
    filtered_img[mask] = 0
    return filtered_img

def nothing(x):
    pass

def stereo_SGBM_tuner(img1, img2):
    win_name = 'window'


    cv2.namedWindow(win_name)
    cv2.createTrackbar("disparity_min", win_name, 20, 10, nothing)
    cv2.createTrackbar("disparity_num", win_name, 20,50, nothing)
    win_size = 5
    min_disp = -1
    max_disp = 63
    num_disp = max_disp - min_disp
    uniqueness_ratio = 5
    block_size = 5

    
    while(1):
        min_disp = cv2.getTrackbarPos("disparity_min", win_name) * 16
        num_disp = cv2.getTrackbarPos("disparity_num", win_name) * 16
        print(num_disp)
        assert(num_disp % 16 is 0)

        stereo_SGBM = cv2.StereoSGBM_create(min_disp, num_disp, block_size)
        disp = stereo_SGBM.compute(img2, img1)

        cv2.imshow(win_name,disp)


        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break

        

    cv2.destroyAllWindows()

if __name__ == '__main__':
    pass
        


