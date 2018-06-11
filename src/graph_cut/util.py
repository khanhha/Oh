import cv2 as cv
import os
import numpy as np
import skimage.filters as skifil
import scipy.ndimage as scifil

def smooth_contour(cnt, sig_ma= 5):
    fil_cnt_0 = scifil.gaussian_filter1d(cnt[:,0,0], sigma=sig_ma)
    fil_cnt_0 = np.reshape(fil_cnt_0, (fil_cnt_0.shape[0], 1, 1))

    fil_cnt_1 = scifil.gaussian_filter1d(cnt[:,0,1], sigma=sig_ma)
    fil_cnt_1 = np.reshape(fil_cnt_1, (fil_cnt_1.shape[0], 1, 1))

    return np.concatenate((fil_cnt_0, fil_cnt_1), axis=2)

def smooth_contour_spline(cnt, order = 5):
    cnt = cv.approxPolyDP(cnt, 1, closed=True)
    fil_cnt_0 = scifil.spline_filter1d(cnt[:,0,0], order=order).astype(np.int32)
    fil_cnt_0 = np.reshape(fil_cnt_0, (fil_cnt_0.shape[0], 1, 1))

    fil_cnt_1 = scifil.spline_filter1d(cnt[:,0,1], order=order).astype(np.int32)
    fil_cnt_1 = np.reshape(fil_cnt_1, (fil_cnt_1.shape[0], 1, 1))

    return np.concatenate((fil_cnt_0, fil_cnt_1), axis=2)

def extract_largest_contour(img_bi):
    im2, contours, hierarchy = cv.findContours(img_bi, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largest_area = -1
    largest_idx = -1
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_idx = i

    return contours[largest_idx]

def scale_image(img, scale):
    size = img.shape[:2]
    size = (int(size[1] * scale), int(size[0] * scale))
    img = cv.resize(img, size, interpolation=cv.INTER_CUBIC)
    return img

def resize_common_size(img):
    return cv.resize(img, (586, 648), interpolation=cv.INTER_CUBIC)

def preprocess_img(img):
    img = cv.medianBlur(img, 5)
    #img = scale_image(img, 0.5)
    return resize_common_size(img)

def remove_light_tubes(img, black = False):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    val, mask = cv.threshold(gray, 190, 255, cv.THRESH_BINARY)
    #mask = cv.dilate(mask, np.ones((3, 3), np.uint8))
    if black:
        img[mask.astype(np.bool)] = 150
    else:
        img = cv.inpaint(img, mask, 10, cv.INPAINT_TELEA)
    return img

def load_ground_truth_silhouette(ground_truth_dir, img_name):
    for name in os.listdir(ground_truth_dir):
        if name in img_name:
            img = cv.imread(f'{ground_truth_dir}/{name}')
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, fg_mask = cv.threshold(img, 1, maxval=255, type=cv.THRESH_BINARY)
            fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (4, 4)))
            fg_mask = resize_common_size(fg_mask)
            return fg_mask

    return None

def color_constancy(img):
    n_pixels = np.prod(img.shape[:2]).astype(np.float)
    img = img.astype(np.float)
    color_means = np.array([0, 0, 0], dtype=np.float)
    color_scales = np.array([0, 0, 0], dtype=np.float)
    for i in range(3):
        color_means[i] = np.mean(img[:, :, i])
    color_scales = np.mean(color_means) / color_means

    for i in range(3):
        img[:, :, i] = color_scales[i] * img[:, :, i]
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)