import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import img_as_float
from skimage.future import graph
from sklearn import mixture
from skimage import exposure
import skimage.filters as filters
import util

def select_best_gmm_model(X, cluster_range = range(5,10)):
    lowest_bic = np.infty
    bic = []
    n_components_range = cluster_range
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

def cluster_GMM(img, mask, n_cluster = None, cov_type=None):
    xs, ys = np.where(mask==True)
    samples = img[xs, ys]
    X_train = np.array(samples)

    if n_cluster == None or cov_type == None:
        clf_fg = select_best_gmm_model(X_train, cluster_range = range(5,15))
        clf_fg.fit(X_train)
    else:
        clf_fg = mixture.GaussianMixture(n_components=n_cluster, covariance_type=cov_type)
        clf_fg.fit(X_train)

    idxs = clf_fg.predict(X_train)

    lablels = np.zeros(mask.shape, np.uint8)
    lablels[xs, ys] = idxs

    return lablels

def estimate_gmm(img, mask, n_cluster = None, cov_type = None):
    samples = img[mask]
    X_train = np.array(samples)

    if n_cluster == None or cov_type == None:
        clf = select_best_gmm_model(X_train, cluster_range = range(5,15))
        clf.fit(X_train)
    else:
        clf = mixture.GaussianMixture(n_components=n_cluster, covariance_type=cov_type)
        clf.fit(X_train)

    return clf


def sk_gmm(img_in, fg_mask, bg_mask, file_name):
    DIR_OUT = 'D:\Projects\Oh\data\images\silhouette\\foreground_backgroud_probability'

    img = img_in

    img = img[:, :, 0:3]

    fg_samples = img[fg_mask]
    bg_samples = img[bg_mask]
    possible_rg = np.bitwise_or(fg_mask, bg_mask)
    possible_rg = np.bitwise_not(possible_rg)

    masked_img = cv.bitwise_and(img_in, img_in, mask=possible_rg.astype(np.uint8))
    segments = slic(masked_img, n_segments=100, compactness=5, convert2lab=True)
    unq_segments = np.unique(segments)
    n_segments = len(unq_segments)

    g = graph.RAG(segments, connectivity=8)

    FG_FLAG = 1
    BG_FLAG = 2
    BG_BDR_FLAG = 3
    POSSIBLE_FLAG = 5
    POSSIBLE_BDR_FLAG = 6

    fg_px_cnt   = np.zeros((n_segments,), dtype=np.uint32)
    bg_px_cnt   = np.zeros((n_segments,), dtype=np.uint32)
    total_cnt   = np.zeros((n_segments,), dtype=np.uint32)
    node_marks  = np.zeros((n_segments,), dtype=np.uint32)

    print("\t create graph")
    for index in np.ndindex(segments.shape):
        current = segments[index]
        fg_px_cnt[current] += int(fg_mask[index])
        bg_px_cnt[current] += int(bg_mask[index])
        total_cnt[current] += 1

    print("\t classify node")
    for node_idx in g:
        fg_percent = float(fg_px_cnt[node_idx])/ float(total_cnt[node_idx])
        bg_percent = float(bg_px_cnt[node_idx])/ float(total_cnt[node_idx])
        possible_cnt = total_cnt[node_idx] - bg_px_cnt[node_idx] - fg_px_cnt[node_idx]
        if bg_percent > 0.5:
            node_marks[node_idx] = BG_FLAG
        elif fg_percent > 0.5:
            node_marks[node_idx] = FG_FLAG
        else:
            node_marks[node_idx] = POSSIBLE_FLAG

    for x, y, in g.edges_iter():
        if node_marks[x] == POSSIBLE_FLAG and node_marks[y] == BG_FLAG:
            node_marks[y] = BG_BDR_FLAG
            #node_marks[x] = POSSIBLE_BDR_FLAG
        elif node_marks[x] == BG_FLAG and node_marks[y] == POSSIBLE_FLAG:
            node_marks[x] = BG_BDR_FLAG
            #node_marks[y] = POSSIBLE_BDR_FLAG

    for node_idx in g:
        if node_marks[node_idx] == BG_BDR_FLAG:
            masked_img[segments==node_idx] = (255,0,0)

    print("\t predict background probability")
    bg_prob = np.zeros(img.shape[:2], np.uint8)
    for node_idx in g:
        if node_marks[node_idx] != POSSIBLE_FLAG:
            continue

        bg_neighbors = []
        for neighbor_idx in g.neighbors(node_idx):
            if node_marks[neighbor_idx] == BG_BDR_FLAG:
                bg_neighbors.append(neighbor_idx)

        if len(bg_neighbors) > 0:
            samples = []
            for bgr_node_idx in bg_neighbors:
                samples.extend(img_in[segments == bgr_node_idx])

            clf = mixture.GaussianMixture(n_components=9, covariance_type='full')
            clf.fit(np.array(samples))

            px_indices = (segments == node_idx)
            scores = clf.score_samples(np.array(img_in[px_indices]))
            mmin = np.min(scores)
            mmax = np.max(scores)
            #print('min: {0}, max: {1}'.format(mmin, mmax))
            scores = (scores - mmin) / (mmax - mmin)
            scores = (scores * 255.0).astype(np.uint8)
            #
            # hist, bins = np.histogram(scores, 256, normed=True)
            # cdf = hist.cumsum()  # cumulative distribution function
            # cdf = 255 * cdf / cdf[-1]  # normalize
            # scores = np.interp(scores, bins[:-1], cdf)

            bg_prob[px_indices] = scores

    plt.figure()
    plt.imshow(mark_boundaries(masked_img, segments))
    plt.savefig(f'{DIR_OUT}\superpixel\{file_name}')
    plt.close()

    plt.figure()
    plt.subplot(1,3,1); plt.imshow(mark_boundaries(img_in, segments))
    plt.subplot(1,3,2); plt.imshow(mark_boundaries(masked_img, segments))
    plt.subplot(1,3,3); plt.imshow(mark_boundaries(bg_prob, segments),cmap='gray')
    plt.savefig(f'{DIR_OUT}\\bg_local_prob\{file_name}', dpi=1000)
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

def load_mean_silhouette(DIR, label):
    for file_name in os.listdir(DIR):
        if label in file_name:
            prob_img = cv.imread(os.path.join(DIR, file_name))
            prob_img = cv.cvtColor(prob_img, cv.COLOR_BGR2GRAY)
            prob_img = util.resize_common_size(prob_img)
            prob_img = np.float32(prob_img) / prob_img.max()
            return prob_img
    return None

def find_mean_silhouette_label(mapping_file, file_name):
    img_lables = mapping_file
    file = open(img_lables)
    for line in file.readlines():
        if file_name in line:
            substr = line.split()
            label = substr[1]
            return label
    return None

def build_fg_bg_masks(prob_map, bg_rect=False, fg_threshold = 0.95, as_bool = False):
    ret, fg_mask = cv.threshold(prob_map, fg_threshold, maxval=1, type=cv.THRESH_BINARY)
    ret, bg_mask = cv.threshold(prob_map, 0, maxval=1, type=cv.THRESH_BINARY)
    fg_mask = (fg_mask * 255).astype(np.uint8)
    bg_mask = (bg_mask * 255).astype(np.uint8)

    kernel_noise = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    bg_mask = cv.morphologyEx(bg_mask, cv.MORPH_OPEN, kernel_noise)  # remove noise
    kernel_bg_dilate = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
    bg_mask = cv.dilate(bg_mask, kernel_bg_dilate)
    bg_mask = cv.morphologyEx(bg_mask, cv.MORPH_CLOSE, kernel_noise)  # remove noise

    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel_noise)  # remove noise
    fg_mask = cv.erode(fg_mask, cv.getStructuringElement(cv.MORPH_RECT, (10, 10)))

    if bg_rect:
        cnts = cv.findContours(bg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv.boundingRect(cnts[0])
        cv.rectangle(bg_mask, (x, y), (x + w, y + h), 255, cv.FILLED)
        # plt.imshow(bg_mask, plt.cm.binary)
        # plt.show()

    bg_mask = 255 - bg_mask
    if as_bool:
        bg_mask = bg_mask.astype(bool)
        fg_mask = fg_mask.astype(bool)

    mask = np.zeros(prob_map.shape[:2], np.uint8)
    mask[:, :] = cv.GC_PR_FGD
    mask[fg_mask] = cv.GC_FGD
    mask[bg_mask] = cv.GC_BGD

    return fg_mask, bg_mask, mask
