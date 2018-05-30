import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
from skimage.measure import compare_ssim
from skimage import data, img_as_float
from sklearn import mixture
from skimage import exposure
import skimage.filters as filters
from scipy.ndimage.filters import generic_filter


def scale_image(img, scale):
    size = img.shape[:2]
    size = (int(size[1] * scale), int(size[0] * scale))
    img = cv.resize(img, size, interpolation=cv.INTER_CUBIC)
    return img


# test color constancy
# dir = "D:\Projects\Oh\data\images\\abcAntony"
# for file_name  in os.listdir(dir):
#     fpath = os.path.join(dir, file_name)
#     if(os.path.isfile(fpath) and 'jpg' in fpath):
#         img = cv.imread(fpath)
#         img_test = img.copy()
#         img = color_constancy(img)
#         plt.subplot(1,2,1); plt.imshow(cv.cvtColor(img_test,cv.COLOR_BGRA2RGB))
#         plt.subplot(1,2,2); plt.imshow(cv.cvtColor(img,cv.COLOR_BGRA2RGB))
#         plt.show()

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


def remove_light_tubes(img, black = False):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    val, mask = cv.threshold(gray, 190, 255, cv.THRESH_BINARY)
    #mask = cv.dilate(mask, np.ones((3, 3), np.uint8))
    if black:
        img[mask.astype(np.bool)] = 150
    else:
        img = cv.inpaint(img, mask, 10, cv.INPAINT_TELEA)
    return img


def create_or_open_dir(dir):
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)


def mean_shift(img):
    img = cv.pyrMeanShiftFiltering(img, 50, 50)
    return img


def subtract_background_skimage(dir, bg_img_path):
    size = (512, 512)
    bg_img = cv.imread(bg_img_path)
    # bg_img = cv.GaussianBlur(bg_img,(5,5),0)
    bg_img = (cv.resize(bg_img, size, cv.INTER_AREA) * 255).astype(np.uint8)
    # bg_img = remove_light_tubes(bg_img)
    bg_img = img_as_float(bg_img)
    # bg_img = cv.cvtColor(bg_img, cv.COLOR_BGR2GRAY)
    fg_scales = {}
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if os.path.isfile(file_path) and '.jpg' in file_name:
            img = cv.imread(file_path)
            img = (cv.resize(img, size, cv.INTER_AREA) * 255).astype(np.uint8)
            # img = remove_light_tubes(img)
            # img = cv.GaussianBlur(img, (5,5),0)
            img = img_as_float(img)
            score, img_dif = compare_ssim(bg_img, img, multichannel=True, data_range=img.max() - img.min(), full=True)
            img_dif = 1.0 - img_dif
            fg_scales[file_path] = img_dif
            img_dif = (img_dif * 255).astype("uint8")
            plt.imshow(img_dif)
            plt.show()

    return fg_scales


def subtract_background_rick(dir, bg_img_path):
    bg_img = cv.imread(bg_img_path)
    bg_img = remove_light_tubes(bg_img)
    bg_img = cv.GaussianBlur(bg_img, (5, 5), 0)
    cv.imwrite(os.path.join("D:\Projects\Oh\data\images\silhouette\grabcut_result\\view_178\\rick_multiplifier",
                            "background.png"), bg_img)
    fg_scales = {}
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if os.path.isfile(file_path) and '.jpg' in file_name:
            img = cv.imread(file_path)
            img = remove_light_tubes(img)
            img = cv.GaussianBlur(img, (5, 5), 0)
            img = cv.resize(img, bg_img.shape[0:2][::-1])
            img_dif = cv.absdiff(bg_img, img)
            img_dif = img_dif / bg_img

            # plt.subplot(1,3,1); plt.imshow(bg_img)
            # plt.subplot(1,3,2); plt.imshow(img)
            # plt.subplot(1,3,3); plt.imshow(img_dif)
            # plt.savefig(os.path.join("D:\Projects\Oh\data\images\silhouette\grabcut_result\\view_178\\rick_multiplifier", file_name))
            # cv.imwrite(os.path.join("D:\Projects\Oh\data\images\silhouette\grabcut_result\\view_178\\rick_multiplifier", file_name), img_dif)
            fg_scales[file_path] = img_dif

    return fg_scales

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

def subtract_background_knn(dir, bg_img_path):
    dir_out = "D:\Projects\Oh\data\images\silhouette\\background_subtractor"
    scale_img_val = 0.5
    bg_img = cv.imread(bg_img_path)
    bg_img = cv.medianBlur(bg_img, 5)
    bg_img = scale_image(bg_img, scale_img_val)
    bg_img = bg_img.astype(np.float32)
    rows,cols = bg_img.shape[:2]

    plt.clf()
    plt.subplot(1, 4, 1)
    plt.imshow(cv.cvtColor(bg_img.astype(np.uint8), cv.COLOR_BGR2RGB), label='input')
    plt.subplot(1, 4, 2)
    plt.imshow(cv.cvtColor(bg_img.astype(np.uint8), cv.COLOR_BGR2RGB), label='cv background subtractor')
    plt.subplot(1, 4, 3)
    plt.imshow(cv.cvtColor(bg_img.astype(np.uint8), cv.COLOR_BGR2RGB), label='morphology open')
    plt.subplot(1, 4, 4)
    plt.imshow(cv.cvtColor(bg_img.astype(np.uint8), cv.COLOR_BGR2RGB), label='contour with largest area')
    plt.savefig(os.path.join(dir_out, 'background_figure.png'), dpi=1000, format='png')
    plt.close()

    # bg_img = remove_light_tubes(bg_img)
    # bg_img = cv.cvtColor(bg_img, cv.COLOR_BGR2GRAY)
    #bg_img = cv.GaussianBlur(bg_img, gau_filter_size, 0)

    range_x = 4
    range_y = 4
    variance = 0.5
    n_noise_samples = 1
    n_bg_samples = (2*range_x ) * (2*range_y) * n_noise_samples
    learning_rate = 1./n_bg_samples

    fgbg = cv.createBackgroundSubtractorKNN(history=n_bg_samples, dist2Threshold=40, detectShadows=True)
    fgbg.apply(bg_img, 1./n_bg_samples)

    Mat = np.float32([[1,0,1],[0,1,1]])
    noise = (variance * np.random.randn(n_noise_samples))
    for x in range(-range_x, range_x):
        for y in range(-range_y, range_y):
            Mat[0,2] = x
            Mat[1,2] = y
            moved_img = cv.warpAffine(bg_img, Mat, (cols, rows), borderMode=cv.BORDER_REFLECT)
            fgbg.apply(moved_img, learning_rate)
    fg_masks = {}
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if os.path.isfile(file_path) and '.jpg' in file_name and file_path not in bg_img_path:
            img = cv.imread(file_path)
            img = cv.medianBlur(img, 5)
            img = scale_image(img, scale_img_val)
            img = img.astype(np.float32)
            # img = remove_light_tubes(img)
            # img = cv.resize(img, size, cv.INTER_AREA)
            # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # img = cv.GaussianBlur(img, gau_filter_size, 0)
            fgmask_0 = fgbg.apply(img, learningRate=0)
            # val, fgmask = cv.threshold(fgmask,  190, 255, cv.THRESH_BINARY)
            val, fgmask = cv.threshold(fgmask_0, 200, 255, cv.THRESH_BINARY)

            strel = cv.getStructuringElement(cv.MORPH_RECT, (18, 18))
            morph_fg_mask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, strel)

            strel = cv.getStructuringElement(cv.MORPH_RECT, (18, 18))
            morph_fg_mask = cv.morphologyEx(morph_fg_mask, cv.MORPH_OPEN, strel)

            # strel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
            # morph_fg_mask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, strel)

            #biggest_contour_img = draw_largest_contour(img.copy(), morph_fg_mask)
            biggest_contour_img = img.copy()

            cv.imwrite(os.path.join(dir_out+'\\after_subtraction', file_name), fgmask)
            cv.imwrite(os.path.join(dir_out+'\\after_healing', file_name), morph_fg_mask)

            img = img.astype(np.uint8)

            plt.clf()
            plt.subplot(1, 4, 1)
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), label='input')
            plt.subplot(1, 4, 2)
            plt.imshow(fgmask_0, cmap='gray', label='cv background subtractor')
            plt.subplot(1, 4, 3)
            plt.imshow(fgmask, cmap='gray', label='morphology open')
            plt.subplot(1, 4, 4)
            plt.imshow(biggest_contour_img, label='contour with largest area')
            plt.savefig(os.path.join(dir_out, file_name), dpi=1000, format='png')
            plt.close()
            # cv.imwrite(os.path.join(dir_out, file_name), fgmask)
            fg_masks[file_path] = fgmask

    return fg_masks


def subtract_background_gmm2(dir, bg_img_path):
    dir_out = "D:\Projects\Oh\data\images\silhouette\\background_subtractor"
    gau_filter_size = (3,3)
    scale_img_val = 0.5
    bg_img = cv.imread(bg_img_path)
    bg_img = cv.medianBlur(bg_img, 5)
    bg_img = scale_image(bg_img, scale_img_val)
    bg_img = bg_img.astype(np.float32)
    rows,cols = bg_img.shape[:2]

    plt.clf()
    plt.subplot(1, 4, 1)
    plt.imshow(cv.cvtColor(bg_img.astype(np.uint8), cv.COLOR_BGR2RGB), label='input')
    plt.subplot(1, 4, 2)
    plt.imshow(cv.cvtColor(bg_img.astype(np.uint8), cv.COLOR_BGR2RGB), label='cv background subtractor')
    plt.subplot(1, 4, 3)
    plt.imshow(cv.cvtColor(bg_img.astype(np.uint8), cv.COLOR_BGR2RGB), label='morphology open')
    plt.subplot(1, 4, 4)
    plt.imshow(cv.cvtColor(bg_img.astype(np.uint8), cv.COLOR_BGR2RGB), label='contour with largest area')
    plt.savefig(os.path.join(dir_out, 'background_figure.png'), dpi=1000, format='png')
    plt.close()

    # bg_img = remove_light_tubes(bg_img)
    # bg_img = cv.cvtColor(bg_img, cv.COLOR_BGR2GRAY)
    #bg_img = cv.GaussianBlur(bg_img, gau_filter_size, 0)

    range_x = 4
    range_y = 4
    variance = 0.5
    n_noise_samples = 1
    n_bg_samples = (2*range_x ) * (2*range_y) * n_noise_samples
    learning_rate = 1./n_bg_samples

    # fgbg = cv.createBackgroundSubtractorKNN(history=100, dist2Threshold=100, detectShadows=True)
    fgbg = cv.createBackgroundSubtractorMOG2(history=n_bg_samples, varThreshold=6, detectShadows=True)
    # fgbg = cv.bgsegm.createBackgroundSubtractorCNT(minPixelStability = 10, useHistory= True, maxPixelStability=200)
    fgbg.apply(bg_img, 1./n_bg_samples)

    Mat = np.float32([[1,0,1],[0,1,1]])
    noise = (variance * np.random.randn(n_noise_samples))
    for x in range(-range_x, range_x):
        for y in range(-range_y, range_y):
            #img_noise = np.random.randint(0, 4, bg_img.shape)
            #img_noise = (img_noise + bg_img).astype(np.uint8)
            Mat[0,2] = x
            Mat[1,2] = y
            moved_img = cv.warpAffine(bg_img, Mat, (cols, rows), borderMode=cv.BORDER_REFLECT)
            fgbg.apply(moved_img, learning_rate)

            # cv.imwrite(f'{dir_out}/debug/{cnt}.png', mask)
            # cnt+=1
            # for j in range(n_noise_samples):
            #     img_noise = (moved_img + noise[j])
            #     mask = fgbg.apply(img_noise, learning_rate)
            #     cv.imwrite(f'{dir_out}/debug/{cnt}.png', mask)
            #     cnt += 1

    fg_masks = {}
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if os.path.isfile(file_path) and '.jpg' in file_name and file_path not in bg_img_path:
            img = cv.imread(file_path)
            img = cv.medianBlur(img, 5)
            img = scale_image(img, scale_img_val)
            img = img.astype(np.float32)
            # img = remove_light_tubes(img)
            # img = cv.resize(img, size, cv.INTER_AREA)
            # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # img = cv.GaussianBlur(img, gau_filter_size, 0)
            fgmask_0 = fgbg.apply(img, learningRate=0)
            # val, fgmask = cv.threshold(fgmask,  190, 255, cv.THRESH_BINARY)
            val, fgmask = cv.threshold(fgmask_0, 200, 255, cv.THRESH_BINARY)

            strel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
            morph_fg_mask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, strel)

            strel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
            morph_fg_mask = cv.morphologyEx(morph_fg_mask, cv.MORPH_OPEN, strel)

            largest_contour = extract_largest_contour(morph_fg_mask)
            morph_fg_mask   = cv.drawContours(morph_fg_mask, [largest_contour], 0, (255,255,255), 2, cv.FILLED)

            strel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
            morph_fg_mask = cv.morphologyEx(morph_fg_mask, cv.MORPH_CLOSE, strel)

            img = img.astype(np.uint8)
            biggest_contour_img = img.copy()

            largest_contour = extract_largest_contour(morph_fg_mask)
            biggest_contour_img = cv.drawContours(img, [largest_contour], 0, (0,255,0), 3)

            cv.imwrite(os.path.join(dir_out+'\\after_subtraction', file_name), fgmask)
            cv.imwrite(os.path.join(dir_out+'\\after_healing', file_name), morph_fg_mask)

            plt.clf()
            plt.subplot(1, 4, 1)
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), label='input')
            plt.subplot(1, 4, 2)
            plt.imshow(fgmask, cmap='gray', label='cv background subtractor')
            plt.subplot(1, 4, 3)
            plt.imshow(morph_fg_mask, cmap='gray', label='morphology open')
            plt.subplot(1, 4, 4)
            plt.imshow(biggest_contour_img, label='contour with largest area')
            plt.savefig(os.path.join(dir_out, file_name), dpi=1000, format='png')
            plt.close()
            # cv.imwrite(os.path.join(dir_out, file_name), fgmask)
            fg_masks[file_path] = fgmask

    return fg_masks

def build_graph_cau_fg_bg_masks(prob_map, bg_rect=False):
    ret, fg_mask = cv.threshold(prob_map, 0.95, maxval=1, type=cv.THRESH_BINARY)
    ret, bg_mask = cv.threshold(prob_map, 0, maxval=1, type=cv.THRESH_BINARY)
    fg_mask = (fg_mask * 255).astype(np.uint8)
    bg_mask = (bg_mask * 255).astype(np.uint8)

    kernel_noise = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    kernel_bg_dilate = cv.getStructuringElement(cv.MORPH_RECT, (25, 25))

    bg_mask = cv.morphologyEx(bg_mask, cv.MORPH_OPEN, kernel_noise)  # remove noise
    bg_mask = cv.dilate(bg_mask, kernel_bg_dilate)
    bg_mask = cv.morphologyEx(bg_mask, cv.MORPH_CLOSE, kernel_noise)  # remove noise

    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel_noise)  # remove noise
    fg_mask = cv.erode(fg_mask, cv.getStructuringElement(cv.MORPH_RECT, (15, 15)))

    if bg_rect:
        cnts = cv.findContours(bg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv.boundingRect(cnts[0])
        cv.rectangle(bg_mask, (x, y), (x + w, y + h), 255, cv.FILLED)
        # plt.imshow(bg_mask, plt.cm.binary)
        # plt.show()

    bg_mask = 255 - bg_mask
    bg_mask = bg_mask.astype(bool)
    fg_mask = fg_mask.astype(bool)

    mask = np.zeros(prob_map.shape[:2], np.uint8)
    mask[:, :] = cv.GC_PR_FGD
    mask[fg_mask] = cv.GC_FGD
    mask[bg_mask] = cv.GC_BGD

    return fg_mask, bg_mask, mask


def pre_process_image(img):
    img = remove_light_tubes(img, True)

    # img = cv.pyrMeanShiftFiltering(img, 10, 10, 1)
    # img = cv.GaussianBlur(img, (5,5), 0)
    # plt.imshow(img)
    # plt.show()

    # img = color_constancy(img)

    # width, height  = img.shape[0], img.shape[1]
    # scale = 0.4
    # win_size = (int(scale * width), int(scale * height))
    # img = exposure.equalize_adapthist(img, kernel_size=win_size, clip_limit=0.03)
    # img = (img * 255).astype(np.uint8)

    return img

def cv_em(img, fg_mask, bg_mask):
    em_fg = cv.ml.EM_create()
    em_bg = cv.ml.EM_create()
    fg_samples = img[fg_mask]
    bg_samples = img[bg_mask]
    possible_rg = np.bitwise_or(fg_mask, bg_mask)
    possible_rg = np.bitwise_not(possible_rg)
    em_fg.trainEM(fg_samples)
    # em_bg.trainEM(bg_samples)
    prob = np.zeros(img.shape[:2], np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if possible_rg[i, j] == True:
                p = em_fg.predict2(img[i, j, :])
                prob[i, j] = np.abs(np.log(np.max(p[1])))
                print(np.max(p[1]), prob[i, j])

    mmin = np.min(prob)
    mmax = np.max(prob)
    prob = (prob - mmin) / (mmax - mmin)
    prob = (255 * prob).astype(np.uint8)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.imshow(prob.astype(np.uint8), cmap=plt.cm.gray, alpha=0.4)
    plt.show()

def select_best_gmm_model(X):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(4, 10)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    print("best component number: {0}".format(gmm.n_components))
    print("best type: {0}".format(gmm.covariance_type))

    return best_gmm

def sk_gmm(img_in, fg_mask, bg_mask, file_name):
    DIR_OUT = 'D:\Projects\Oh\data\images\silhouette\\foreground_backgroud_probability'

    img = img_in

    img = img[:, :, 0:3]

    fg_samples = img[fg_mask]
    bg_samples = img[bg_mask]
    possible_rg = np.bitwise_or(fg_mask, bg_mask)
    possible_rg = np.bitwise_not(possible_rg)

    masked_img = cv.bitwise_and(img_in, img_in, mask=possible_rg.astype(np.uint8))
    segments = slic(masked_img, n_segments=1500, convert2lab=True)
    plt.figure()
    plt.imshow(mark_boundaries(masked_img, segments))
    plt.savefig(f'{DIR_OUT}\superpixel\{file_name}')
    plt.close()
    return

    X_train_fg = np.array(fg_samples)
    X_train_bg = np.array(bg_samples)

    print('training foreground...')
    #clf_fg = select_best_gmm_model(X_train_fg)
    clf_fg = mixture.GaussianMixture(n_components=9, covariance_type='full')
    clf_fg.fit(X_train_fg)

    print('training background...')
    #clf_bg = select_best_gmm_model(X_train_bg)
    clf_bg = mixture.GaussianMixture(n_components=9, covariance_type='full')
    clf_bg.fit(X_train_bg)

    print('collecting pixels in possible area...')
    prob_img_fg = np.zeros(img.shape[:2], np.float32)
    prob_img_bg = np.empty_like(prob_img_fg)
    pred_idx = list()
    pred_samples = list()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if possible_rg[i, j] == True:
                pred_idx.append((i, j))
                pred_samples.append(img[i, j])

    print('predicting probabilities...')
    X_pred = np.array(pred_samples)
    scores_fg = clf_fg.score_samples(X_pred)
    scores_bg = clf_bg.score_samples(X_pred)
    for i, idx in enumerate(pred_idx):
        prob_img_fg[idx] = scores_fg[i]
        prob_img_bg[idx] = scores_bg[i]

    mmin = np.min(scores_fg)
    mmax = np.max(scores_fg)
    print('min: {0}, max: {1}'.format(mmin, mmax))
    prob_fg = (prob_img_fg - mmin) / (mmax - mmin)
    prob_fg = (prob_fg*255.0).astype(np.uint8)
    prob_fg = cv.bitwise_and(prob_fg, prob_fg, mask=possible_rg.astype(np.uint8))
    prob_adj_fg = exposure.adjust_gamma(prob_fg, 4)

    mmin = np.min(scores_bg)
    mmax = np.max(scores_bg)
    print('min: {0}, max: {1}'.format(mmin, mmax))
    prob_bg = (prob_img_bg - mmin) / (mmax - mmin)
    prob_bg = (prob_bg*255.0).astype(np.uint8)
    prob_bg = cv.bitwise_and(prob_bg, prob_bg, mask=possible_rg.astype(np.uint8))
    prob_adj_bg = exposure.adjust_gamma(prob_bg, 4)

    # plt.subplot(141);
    # plt.imshow(prob_fg, cmap='gray');
    # plt.title("log probability")
    # plt.subplot(142);
    # plt.imshow(prob_adj_fg, cmap='gray');
    # plt.subplot(143);
    # plt.imshow(prob_bg, cmap='gray');
    # plt.subplot(144);
    # plt.imshow(prob_adj_bg, cmap='gray');
    # plt.title("gamma adjusted")
    # plt.show()

    possible_rg = possible_rg.astype(np.uint8)
    prob_adj_fg = img_as_float(prob_adj_fg)
    prob_adj_bg = img_as_float(prob_adj_bg)
    prob_mix = prob_adj_fg * (1.0-prob_adj_bg)
    prob_mix = cv.bitwise_and(prob_mix, prob_mix, mask=possible_rg)

    # masked_img =cv.bitwise_and(img_in, img_in, mask=possible_rg)
    # segments = slic(masked_img, n_segments=2000, convert2lab=True)
    # pbl_segments_ids = np.unique(segments[possible_rg[:,:]>0])
    # avg_seg_probs = np.array(
    #     [np.mean(prob_mix[np.bitwise_and(segments==id, possible_rg.astype(np.bool))]) for id in pbl_segments_ids])
    #
    # avg_prob_mix = np.empty_like(prob_mix)
    # for i, seg_id in enumerate(pbl_segments_ids):
    #     idxs = (segments==seg_id)
    #     avg_prob_mix[idxs] = avg_seg_probs[i]

    # plt.figure()
    # plt.imshow(mark_boundaries(masked_img, segments))
    # plt.show()

    prob_mix = (prob_mix*255).astype(np.uint8)
    new_mask = prob_mix > 0
    #avg_prob_mix = (avg_prob_mix*255).astype(np.uint8)
    #avg_prob_mix = cv.bitwise_and(avg_prob_mix, avg_prob_mix, mask=possible_rg)
    edge = filters.sobel(prob_mix, new_mask)

    plt.figure()
    plt.subplot(131), plt.imshow(cv.cvtColor(img_in, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.imshow(prob_mix, cmap='gray')
    plt.subplot(133), plt.imshow(edge, cmap='gray')
    plt.savefig(f'{DIR_OUT}\{file_name}', dpi=1000)
    plt.close()

    masked_img =cv.bitwise_and(img_in, img_in, mask=possible_rg)
    segments = slic(masked_img, n_segments=200, convert2lab=True)
    superpixel = mark_boundaries(masked_img, segments)
    cv.imwrite(f'{DIR_OUT}\\superpixel\{file_name}', superpixel)

dir = "D:\Projects\Oh\data\images\silhouette\images\\view_178"
bg_img_path = "D:\Projects\Oh\data\images\\trainim_LL_0428\\test\\image178_empty_LL_0321.jpg"
# fg_scales = subtract_background_gmm2(dir, bg_img_path)
# fg_scales = subtract_background_rick(dir, bg_img_path)
# fg_scales = subtract_background_skimage(dir, bg_img_path)

img_lables = 'D:\Projects\Oh\data\images\silhouette\mapping_image_avg_silhouette.txt'
file = open(img_lables)
label_imgs = {}
for line in file.readlines():
    substr = line.split()
    label = substr[1]
    if label not in label_imgs:
        label_imgs[label] = list()
    label_imgs[label].append(substr[0])

prob_map_dir = 'D:\Projects\Oh\data\images\silhouette\\avg_silhouette_map'
prob_imgs = {}
for file_name in os.listdir(prob_map_dir):
    label = os.path.splitext(file_name)[0]  # extract label file name
    label = os.path.splitext(label)[0]  # remove extension .png, .jpg
    prob_img = cv.imread(os.path.join(prob_map_dir, file_name))
    prob_img = cv.cvtColor(prob_img, cv.COLOR_BGR2GRAY)
    prob_img = np.float32(prob_img) / prob_img.max()
    #prob_img = scale_image(prob_img, 0.5)
    prob_imgs[label] = prob_img

out_dir = 'D:\Projects\Oh\data\images\silhouette\grabcut_result'
for label, img_paths in label_imgs.items():
    prob_map = prob_imgs[label]
    dir = os.path.join(out_dir, str(label))
    create_or_open_dir(dir)
    for img_path in img_paths:
        file_name = os.path.basename(img_path)
        #if file_name not in 'image178_laxsquadT_lBlue_LL_0427.jpg':
        #    continue
        print("processing file {0}".format(img_path))
        img = cv.imread(img_path)
        img = cv.medianBlur(img, 5)
        img = cv.resize(img, (prob_img.shape[1], prob_img.shape[0]), cv.INTER_CUBIC)
        img = pre_process_image(img)
        org_img = img.copy()
        # if img_path in fg_scales:
        #    fg_scale = fg_scales[img_path]
        #    fg_scale = cv.resize(fg_scale, prob_img.shape[0:2][::-1])
        #    img = (fg_scale * img).astype(np.uint8)
        # plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        # plt.imshow(cv.cvtColor(fg_mask, cv.COLOR_BGR2RGB))
        # plt.show()

        fg_mask, bg_mask, mask = build_graph_cau_fg_bg_masks(prob_map, False)

        # cv_em(img, fg_mask, bg_mask)
        sk_gmm(img, fg_mask, bg_mask, file_name)
        continue

        img2 = img.copy()
        for i in range(2):
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            cv.grabCut(img2, mask, None, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)

        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        output = cv.bitwise_and(org_img, org_img, mask=mask2)

        plt.subplot(1, 4, 1);
        plt.imshow(prob_map)
        plt.subplot(1, 4, 2);
        plt.imshow(img)
        plt.subplot(1, 4, 3);
        plt.imshow(img)
        plt.imshow(fg_mask, alpha=0.3, cmap='cool')
        plt.imshow(bg_mask, alpha=0.3, cmap='Wistia')
        plt.subplot(1, 4, 4);
        plt.imshow(output)
        # plt.show()

        file_name = file_name.rpartition('.')[-3]  # get rid of extension

        img_file_name = "{0}.png".format(file_name)
        fig_file_name = "{0}_figure.png".format(file_name)

        plt.savefig(os.path.join(dir, fig_file_name), dpi=1000)
        plt.close()

        plt.imshow(org_img)
        plt.imshow(mask2, alpha=0.4, cmap='gray')
        plt.savefig(os.path.join(dir, img_file_name), dpi=1000)
        plt.close()
        # cv.imwrite(os.path.join(dir, img_file_name),output)
