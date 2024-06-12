import cv2
import numpy as np
from scipy import ndimage
import math
import argparse
import os
import all_functions_used as helpers

def predict(imagepath, output_dir):
    img = helpers.load_image(imagepath)
    print(img.shape)
    org = helpers.remove_noise_and_preprocess(img)
    new_img = helpers.preprocess(img)
    
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            if org[i][j] == 255 and new_img[i][j] == 255:
                continue
            else:
                new_img[i][j] = 0

    cv2.imshow("processed_image", new_img)
    cv2.waitKey(1000)

    x1, x2, y1, y2 = helpers.houghtransform(new_img)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    rot = ndimage.rotate(new_img, angle)
    rot = helpers.word_segmentation(rot)
    cv2.imshow('Rotated_word_image', rot)
    cv2.waitKey(1000)
    
    dilated = rot.copy()
    start_char = []
    end_char = []
    
    row = np.zeros(dilated.shape[1])
    mxrow = 0
    mxcnt = 0
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(dilated, kernel, iterations=1)
    dilated = cv2.erode(dilated, kernel, iterations=1)
    
    for i in range(dilated.shape[0]):
        cnt = 0
        for j in range(dilated.shape[1]):
            if dilated[i][j] == 255:
                cnt += 1
        if mxcnt < cnt:
            mxcnt = cnt
            mxrow = i
            
    print(dilated.shape[0])
    plus = dilated.shape[0] // 10
    for i in range(0, mxrow + plus):
        dilated[i] = row
        
    cv2.imshow("HeaderLine Removed", dilated)
    cv2.waitKey(1000)
    
    col_sum = np.zeros((dilated.shape[1]))
    col_sum = np.sum(dilated, axis=0)
    thresh = (0.08 * dilated.shape[0])
    
    for i in range(1, dilated.shape[1]):
        if col_sum[i - 1] <= thresh and col_sum[i] > thresh and col_sum[i + 1] > thresh:
            start_char.append(i)
        elif col_sum[i - 1] > thresh and col_sum[i] <= thresh and col_sum[i + 1] <= thresh:
            end_char.append(i)
            
    start_char.append(end_char[-1])
    character = []
    
    for i in range(1, len(start_char)):
        roi = rot[:, start_char[i - 1]:start_char[i]]
        h = roi.shape[1]
        w = roi.shape[0]
        roi = helpers.extractroi(roi)
        roi = cv2.resize(roi, (180, 180))
        if helpers.check(roi) and h >= 30 and w >= 30:
            character.append(roi)
            cv2.imshow('CHARACTER_SEGMENTED', roi)
            cv2.waitKey(1000)
            char_img_path = os.path.join(output_dir, f'char_{i}.png')
            cv2.imwrite(char_img_path, roi)

    return character

def test():
    image_paths = ['D:\Handwritten-Hindi-Word-Recognition\hindi.png']
    output_dir = './segmented_characters/'
    os.makedirs(output_dir, exist_ok=True)
    
    for image_path in image_paths:
        image = cv2.imread(image_path)
        segmented_characters = predict(image_path, output_dir)
        print('Segmented characters saved in', output_dir)

if __name__ == "__main__":
    test()
