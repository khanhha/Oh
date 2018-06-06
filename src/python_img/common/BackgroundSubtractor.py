import cv2 as cv
import numpy as np
import scipy.ndimage as scifil
import  matplotlib.pyplot as plt

class BackgroundSubtractor:

    def __init__(self, bg_img_path):
        self._bg_img = cv.imread(bg_img_path)
        self._bg_img = self.__resize_common_size(self._bg_img)
        self.__train_model()

    def extract_foreground_mask(self, img_in):
        img = self.__resize_common_size(img_in)
        img = img.astype(np.float32)
        fgmask_0 = self._bg_model.apply(img, learningRate=0)

        val, fgmask = cv.threshold(fgmask_0, 200, 255, cv.THRESH_BINARY)

        strel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        morph_fg_mask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, strel)

        strel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
        morph_fg_mask = cv.morphologyEx(morph_fg_mask, cv.MORPH_OPEN, strel)

        largest_contour = self.__extract_largest_contour(morph_fg_mask)
        morph_fg_mask = cv.drawContours(morph_fg_mask, [largest_contour], 0, (255, 255, 255), 2, cv.FILLED)

        strel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
        morph_fg_mask = cv.morphologyEx(morph_fg_mask, cv.MORPH_CLOSE, strel)

        largest_contour = self.__extract_largest_contour(morph_fg_mask)
        largest_contour = self.__smooth_contour(largest_contour, 3)

        out_mask = np.zeros_like(fgmask, dtype=np.uint8)
        out_mask = cv.drawContours(out_mask, [largest_contour], 0, (255, 255, 255), cv.FILLED)

        out_mask = cv.resize(out_mask, (img_in.shape[1], img_in.shape[0]), interpolation=cv.INTER_NEAREST)

        out_img = cv.bitwise_and(img_in, img_in, mask=out_mask)

        return  out_img

    def __train_model(self):
        self._bg_img = self._bg_img.astype(np.float32)
        rows, cols = self._bg_img.shape[:2]

        range_x = 4
        range_y = 4
        n_noise_samples = 1
        n_bg_samples = (2 * range_x) * (2 * range_y) * n_noise_samples
        learning_rate = 1. / n_bg_samples

        self._bg_model= cv.createBackgroundSubtractorMOG2(history=n_bg_samples, varThreshold=6, detectShadows=True)
        self._bg_model.apply(self._bg_img, 1. / n_bg_samples)

        Mat = np.float32([[1, 0, 1], [0, 1, 1]])
        for x in range(-range_x, range_x):
            for y in range(-range_y, range_y):
                # img_noise = np.random.randint(0, 4, bg_img.shape)
                # img_noise = (img_noise + bg_img).astype(np.uint8)
                Mat[0, 2] = x
                Mat[1, 2] = y
                moved_img = cv.warpAffine(self._bg_img, Mat, (cols, rows), borderMode=cv.BORDER_REFLECT)
                self._bg_model.apply(moved_img, learning_rate)

        return self._bg_model

    def __extract_largest_contour(self, img_bi):
        im2, contours, hierarchy = cv.findContours(img_bi, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        largest_area = -1
        largest_idx = -1
        for i, contour in enumerate(contours):
            area = cv.contourArea(contour)
            if area > largest_area:
                largest_area = area
                largest_idx = i

        return contours[largest_idx]

    def __resize_common_size(self, img):
        return cv.resize(img, (int(1944), int(2592)), interpolation=cv.INTER_CUBIC)

    def __smooth_contour(self, cnt, sig_ma=5):
        fil_cnt_0 = scifil.gaussian_filter1d(cnt[:, 0, 0], sigma=sig_ma)
        fil_cnt_0 = np.reshape(fil_cnt_0, (fil_cnt_0.shape[0], 1, 1))

        fil_cnt_1 = scifil.gaussian_filter1d(cnt[:, 0, 1], sigma=sig_ma)
        fil_cnt_1 = np.reshape(fil_cnt_1, (fil_cnt_1.shape[0], 1, 1))

        return np.concatenate((fil_cnt_0, fil_cnt_1), axis=2)

    def remove_light_tubes(img, black=False):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        val, mask = cv.threshold(gray, 190, 255, cv.THRESH_BINARY)
        if black:
            img[mask.astype(np.bool)] = 150
        else:
            img = cv.inpaint(img, mask, 10, cv.INPAINT_TELEA)

        return img