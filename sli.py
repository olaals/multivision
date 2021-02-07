
import numpy as np
import cv2




def create_gray_code_pattern(pattern_number, image_width, image_height, channels=3):
    divisions = 2**(pattern_number)
    step = int(image_width / divisions)

    if channels == 3:
        img = np.zeros((image_height, image_width, channels), dtype=np.uint8)
        for i in range(divisions):
            if i % 2 ==0:
                img[:,i*step:(i+1)*step,:]=255
    if channels == 1:
        img = np.zeros((image_height, image_width), dtype=np.uint8)
        for i in range(divisions):
            if i % 2 ==0:
                img[:,i*step:(i+1)*step]=255
    return img

def create_rainbow_pattern_img(patterns, image_width, image_height):
    img = np.zeros((image_height, image_width, 3), np.uint8)
    divisions = 2 ** (patterns+1)
    step = int(image_width / divisions)
    period = 2 * step
    for i in range(divisions):
        if i % 2 == 0:
            continue
        if i % 6 == 1:
            img[:, i*step:(i+1)*step, 0] = 255
        if i % 6 == 3:
            img[:, i*step:(i+1)*step, 1] = 255
        if i % 6 == 5:
            img[:, i*step:(i+1)*step, 2] = 255
    return img




if __name__ == '__main__':
    img = create_gray_code_pattern(1, 1200,1000, channels=3)
    cv2.imshow("sfdd", img)
    cv2.waitKey(0)
    img2 = create_rainbow_pattern_img(2, 1000, 1000)
    cv2.imshow("sffd3", img2)
    cv2.waitKey(0)


