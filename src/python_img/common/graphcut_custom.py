import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.segmentation as skseg
import skimage.filters as skfilter
import skimage.morphology as skmorph
import skimage.feature as skfeature
import skimage.exposure as skexposure
from skimage.future import graph
import skimage.color as skcolor
import scipy.ndimage as ndi
import maxflow

import prob_map
import util
from BackgroundSubtractor import BackgroundSubtractor

FG_NODE = 1
BG_NODE = 2
BG_BDR_FLAG = 3
MID_NODE = 5
POSSIBLE_BDR_FLAG = 6

FONT_SIZE = 3

DIR = "D:/Projects/Oh/data/images/silhouette/images/view_178/"
GROUND_TRUTH_DIR = 'D:/Projects/Oh/data/images/silhouette/silhouette/view_178/'
PROB_MAP_DIR = 'D:/Projects/Oh/data/images/silhouette/avg_silhouette_map/'
OUT_DIR = 'D:/Projects/Oh/data/images/silhouette/custom_grabcut/latest_result/'
OUT_DIR_TEST = 'D:/Projects/Oh/data/images/silhouette/custom_grabcut/test/'
OUT_DIR_EDGE_MAP = 'D:/Projects/Oh/data/images/silhouette/custom_grabcut/edge_map/'
OUT_DIR_PROB_MAP = 'D:/Projects/Oh/data/images/silhouette/custom_grabcut/prob_map/'
img_label_mapping = 'D:\Projects\Oh\data\images\silhouette\mapping_image_avg_silhouette.txt'
bg_img_path = "D:\Projects\Oh\data\images\\trainim_LL_0428\\test\\image178_empty_LL_0321.jpg"
exp_type = 'bg_sub_'

CUR_FILENAME = ''

def superpixel_graph(img, fg_mask, bg_mask, edge_map):
    masked_img = cv.bitwise_and(img, img, mask=mid_mask.astype(np.uint8))
    #masked_img = img

    segments = skseg.slic(masked_img, n_segments=15000, convert2lab=True)
    unq_segments = np.unique(segments)
    n_segments = len(unq_segments)

    g = graph.rag_boundary(segments, edge_map)

    # lc = graph.show_rag(segments, g, edges, img_cmap='gray', edge_cmap='viridis', edge_width=1.2)
    # plt.colorbar(lc, fraction=0.03)
    # skio.show()

    fg_px_cnt   = np.zeros((n_segments,), dtype=np.uint32)
    bg_px_cnt   = np.zeros((n_segments,), dtype=np.uint32)
    total_cnt   = np.zeros((n_segments,), dtype=np.uint32)
    node_marks  = np.zeros((n_segments,), dtype=np.uint32)

    for index in np.ndindex(segments.shape):
        current = segments[index]
        fg_px_cnt[current] += int(fg_mask[index])
        bg_px_cnt[current] += int(bg_mask[index])
        total_cnt[current] += 1

    for node_idx in g:
        fg_percent = float(fg_px_cnt[node_idx])/ float(total_cnt[node_idx])
        bg_percent = float(bg_px_cnt[node_idx])/ float(total_cnt[node_idx])
        possible_cnt = total_cnt[node_idx] - bg_px_cnt[node_idx] - fg_px_cnt[node_idx]
        if bg_percent > 0.9:
            node_marks[node_idx] = BG_NODE
        elif fg_percent > 0.9:
            node_marks[node_idx] = FG_NODE
        else:
            node_marks[node_idx] = MID_NODE

    return g, node_marks, segments

def blend_log_prob_map(indices, fg_map, bg_map, bool_img, alpha = 0.5):
    bi_bg_img = np.bitwise_not(bool_img)
    bi_fg_img = np.bitwise_not(skmorph.binary_dilation(bi_bg_img, skmorph.disk(3)))
    bi_bg_img = skmorph.binary_erosion(bi_bg_img, skmorph.disk(10))

    bg_map_1 = skfilter.gaussian(bi_bg_img.astype(np.float32), 1)
    fg_map_1 = skfilter.gaussian(bi_fg_img.astype(np.float32), 1)

    #fg_map[indices[0],indices[1]] = alpha * fg_map[indices[0],indices[1]] + (1.0-alpha)*fg_map_1[indices[0],indices[1]]
    #bg_map[indices[0],indices[1]] = alpha * bg_map[indices[0],indices[1]] + (1.0-alpha)*bg_map_1[indices[0],indices[1]]

    fg_map[indices[0], indices[1]] = bool_img.astype(np.float32)[indices[0], indices[1]]
    bg_map[indices[0], indices[1]] = np.bitwise_not(bool_img).astype(np.float32)[indices[0], indices[1]]

    return fg_map, bg_map

def rescale_img(img, indices, out_range=(0.0,1.0)):
    minval = img[indices[0], indices[1]].min()
    maxval = img[indices[0], indices[1]].max()
    img[mid_rows, mid_cols] = out_range[0] + (out_range[1] - out_range[0]) * (img[mid_rows, mid_cols] - minval)/(maxval-minval)
    return img

def output_prob_segment(segments, output_segments, fg_probs, bg_probs):
    fg_img = np.zeros_like(segments, dtype=np.float32)
    bg_img = np.zeros_like(segments, dtype=np.float32)
    for i, seg in enumerate(output_segments):
        fg_img[segments==seg] = fg_probs[i]
        bg_img[segments==seg] = bg_probs[i]
    return np.hstack((fg_img, bg_img))

def set_segment_value(all_segments, segments, values, out_range = (0,1)):
    value_img = np.zeros_like(all_segments, dtype=np.float32)
    values = out_range[0] + (out_range[1]-out_range[0]) * (values - values.min()) / (values.max() - values.min())
    for i, seg in enumerate(segments):
        value_img[all_segments==seg] = values[i]
    return value_img

def calc_prob_images(img, fg_mask, bg_mask, mid_mask_not, mid_mask_idxs):
    bg_mask_dilated = skmorph.binary_dilation(np.bitwise_not(bg_mask), skmorph.rectangle(50,50))
    bg_train_sample_mask = np.bitwise_and(bg_mask_dilated, bg_mask)
    gmm_bg = prob_map.estimate_gmm(img, bg_train_sample_mask, 9, 'full')
    gmm_fg = prob_map.estimate_gmm(img, fg_mask, 9, 'full')

    mid_samples = img[mid_mask_idxs[0], mid_mask_idxs[1]]

    scores_fg_img = np.zeros_like(fg_mask, dtype=np.float32)
    scores_bg_img = np.zeros_like(fg_mask, dtype=np.float32)

    scores_fg_img[mid_mask_idxs[0], mid_mask_idxs[1]] = gmm_fg.score_samples(mid_samples)
    scores_bg_img[mid_mask_idxs[0], mid_mask_idxs[1]] = gmm_bg.score_samples(mid_samples)

    scores_fg_img = ndi.median_filter(scores_fg_img, size=5)
    scores_bg_img = ndi.median_filter(scores_bg_img, size=5)

    scores_fg_img[mid_mask_not] = 0
    scores_bg_img[mid_mask_not] = 0

    return scores_fg_img, scores_bg_img

def calc_prob_map_superpixel_segment(scores_fg_img, scores_bg_img, segments, mid_segments, mid_mask_idxs):
    scores_seg_fg = np.array([np.median(scores_fg_img[segments == idx], axis=0) for idx in mid_segments])
    scores_seg_bg = np.array([np.median(scores_bg_img[segments == idx], axis=0) for idx in mid_segments])

    scores_fg_seg_img = set_segment_value(segments, mid_segments, scores_seg_fg)
    scores_bg_seg_img = set_segment_value(segments, mid_segments, scores_seg_bg)

    plt.clf()
    plt.subplot(221), plt.imshow(rescale_img(scores_fg_img, mid_mask_idxs, (0.0, 1.0)), cmap='gray'), plt.gca().set_title('foreground prob per pixel', fontsize= FONT_SIZE)
    plt.subplot(222), plt.imshow(rescale_img(scores_bg_img, mid_mask_idxs, (0.0, 1.0)), cmap='gray'), plt.gca().set_title('background prob per pixel', fontsize= FONT_SIZE)
    plt.subplot(223), plt.imshow(scores_fg_seg_img, cmap='gray'), plt.gca().set_title('foreground prob per segment: median filter', fontsize= FONT_SIZE)
    plt.subplot(224), plt.imshow(scores_bg_seg_img, cmap='gray'), plt.gca().set_title('foreground prob per segment: median filter', fontsize= FONT_SIZE)

    plt.savefig(f'{OUT_DIR_PROB_MAP}{CUR_FILENAME}.png', dpi = 1000)

    return scores_seg_fg, scores_seg_bg

def calc_edge_map(img, mask):
    img_1 = img[:,:,::-1]
    edge_map = skfilter.sobel(skcolor.rgb2gray(img_1), mask= mask)
    #edge_map = skexposure.equalize_adapthist(edge_map, kernel_size=30, clip_limit=1.0)
    edge_map = skexposure.equalize_adapthist(edge_map, kernel_size=30)
    #edge_map = ndi.median_filter(edge_map, size=5)
    edge_map[np.bitwise_not(mask)] = 0
    return edge_map

def suplement_edge_map_with_prob_maps(edge_map, mid_mask, fg_prob_map, bg_prob_map):
    fg_edge_map = skfilter.sobel(fg_prob_map, mask = mid_mask)
    bg_edge_map = skfilter.sobel(bg_prob_map, mask = mid_mask)

    blended_edge_map = 0.4 *edge_map + 0.3 * fg_edge_map + 0.3 * bg_edge_map
    plt.clf()
    plt.subplot(231); plt.imshow(edge_map),    plt.gca().set_title('sobel edge from input img + increase local contrast', fontsize= FONT_SIZE)
    plt.subplot(232); plt.imshow(fg_edge_map), plt.gca().set_title('sobel edge from foreground prob map', fontsize= FONT_SIZE)
    plt.subplot(233); plt.imshow(bg_edge_map), plt.gca().set_title('sobel edge from background prob map', fontsize= FONT_SIZE)
    plt.subplot(235); plt.imshow(blended_edge_map), plt.gca().set_title('blended edge: 0.4*e0 + 0.3*e1 + 0.3*e2', fontsize= FONT_SIZE)
    plt.savefig(f'{OUT_DIR_EDGE_MAP}{CUR_FILENAME}.png', dpi=1000)

    return blended_edge_map

#bgsub = BackgroundSubtractor(bg_img_path)

plt.rcParams['xtick.labelsize'] = 2
plt.rcParams['ytick.labelsize'] = 2

for file_name in os.listdir(DIR):
    #if file_name not in 'image178_laxsquadT_mBBlue_LL_0427.jpg':
    #    continue
    CUR_FILENAME = file_name[0:-4]

    print(f'processing file {file_name}')
    img = util.preprocess_img(cv.imread(f'{DIR}{file_name}'))

    print('\tcalculate silhouette')
    prb_label = prob_map.find_mean_silhouette_label(img_label_mapping, file_name)
    prb_img = prob_map.load_mean_silhouette(PROB_MAP_DIR, prb_label)

    fg_mask, bg_mask, mask = prob_map.build_fg_bg_masks(prb_img, False, fg_threshold = 0.98, as_bool = True)
    mid_mask = np.bitwise_and(np.bitwise_not(fg_mask), np.bitwise_not(bg_mask))
    mid_mask_not = np.bitwise_not(mid_mask)
    mid_rows, mid_cols = np.where(mid_mask==True)
    mid_mask_idxs = (mid_rows, mid_cols)

    # print('\tbackground subtraction mask')
    # _, fg_mask_bgsub = bgsub.extract_foreground_mask(img)
    #
    # fg_mask_bgsub = np.bitwise_or(fg_mask, fg_mask_bgsub)
    # fg_prob_bgsub = skmorph.binary_erosion(fg_mask_bgsub, skmorph.disk(10))
    # fg_prob_bgsub = skfilter.gaussian(fg_prob_bgsub.astype(np.float32), 15)
    #
    # bg_prob_bgsub = np.bitwise_not(skmorph.binary_dilation(fg_mask_bgsub, skmorph.disk(10)))
    # bg_prob_bgsub = skfilter.gaussian(bg_prob_bgsub.astype(np.float32), 15)

    # test
    # plt.subplot(131); plt.imshow(img); plt.imshow(skmorph.binary_erosion(fg_mask_bgsub, skmorph.disk(10)), alpha=0.5)
    # plt.subplot(132); plt.imshow(img); plt.imshow(fg_prob_bgsub, alpha=0.5)
    # plt.subplot(133); plt.imshow(img); plt.imshow(bg_prob_bgsub, alpha=0.5)
    # plt.show()
    # continue

    #fg_mask = np.bitwise_or(fg_mask, skmorph.binary_erosion(fg_mask_bgsub, skmorph.disk(20)))

    print('\tedge map')
    img = util.remove_light_tubes(img, black=True)
    edge_map = calc_edge_map(img, mid_mask)

    print('\tsuperpixel graph')
    graph_segments, node_marks, segments = superpixel_graph(img, fg_mask, bg_mask, edge_map)
    mid_segments_mask = (node_marks==MID_NODE)
    mid_segments_mapping = np.cumsum(mid_segments_mask.astype(np.uint32))
    mid_segments = np.where(mid_segments_mask==True)[0]

    print('\testimate probability map')
    scores_fg_img, scores_bg_img = calc_prob_images(img, fg_mask, bg_mask, mid_mask_not, mid_mask_idxs)

    #scores_fg_img, fg_minval, fg_maxval = rescale_img(scores_fg_img, (mid_rows, mid_cols), (0,1))
    #scores_bg_img, bg_minval, bg_maxval = rescale_img(scores_bg_img, (mid_rows, mid_cols), (0,1))
    #ground_truth_sil = util.load_ground_truth_silhouette(GROUND_TRUTH_DIR, file_name)
    #scores_fg_img, scores_bg_img = blend_log_prob_map((mid_rows, mid_cols), scores_fg_img, scores_bg_img, ground_truth_sil.astype(np.bool))
    #scores_fg_img[mid_rows, mid_cols] = fg_minval + (fg_maxval - fg_minval)* scores_fg_img[mid_rows, mid_cols]
    #scores_bg_img[mid_rows, mid_cols] = bg_minval + (bg_maxval - bg_minval)* scores_bg_img[mid_rows, mid_cols]

    #scores_seg_fg = np.array([np.median(scores_fg_img[segments == idx], axis=0) for idx in mid_segments])
    #scores_seg_bg = np.array([np.median(scores_bg_img[segments == idx], axis=0) for idx in mid_segments])
    scores_seg_fg, scores_seg_bg = calc_prob_map_superpixel_segment(scores_fg_img, scores_bg_img, segments, mid_segments, mid_mask_idxs)

    #test
    # test_fg_img = np.zeros_like(segments, dtype=np.float32)
    # test_bg_img = np.zeros_like(segments, dtype=np.float32)
    # for i, seg in enumerate(mid_segments):
    #     test_fg_img[segments==seg] = scores_fg[i]
    #     test_bg_img[segments==seg] = scores_bg[i]
    # test_fg_img = skseg.mark_boundaries(test_fg_img, segments)
    # test_bg_img = skseg.mark_boundaries(test_bg_img, segments)
    # plt.subplot(121), plt.imshow(img[:,:,::-1]); plt.imshow(test_fg_img, cmap='gray', alpha = 0.8)
    # plt.subplot(122), plt.imshow(img[:,:,::-1]); plt.imshow(test_bg_img, cmap='gray', alpha = 0.8)
    # plt.savefig(f'{OUT_DIR_TEST}/{file_name[0:-4]}_prob.png', dpi = 1000)
    # continue

    edge_map = suplement_edge_map_with_prob_maps(edge_map, mid_mask, rescale_img(scores_fg_img,(mid_rows, mid_cols)), rescale_img(scores_bg_img,(mid_rows, mid_cols)))

    print('\tmax flow')
    num_nodes = np.prod(mid_segments.shape)
    graph_maxflow = maxflow.Graph[float](num_nodes, num_nodes*4)
    nodes = graph_maxflow.add_nodes(num_nodes)

    smoothess_values = []
    for x, y, d in graph_segments.edges_iter(data=True):
        if mid_segments_mask[x] == True and mid_segments_mask[y] == True:
            mid_node_idx_x = nodes[mid_segments_mapping[x]-1]
            mid_node_idx_y = nodes[mid_segments_mapping[y]-1]
            smoothess_values.append(d['weight'])
            smoothess_cost = d['weight']
            graph_maxflow.add_edge(nodes[mid_node_idx_x], nodes[mid_node_idx_y], smoothess_cost, smoothess_cost)

    print(f'\tsmoothess  min, max = {min(smoothess_values)}, {max(smoothess_values)}')
    print(f'\tforeground min, max = {scores_seg_fg.min()}, {scores_seg_fg.max()}')
    print(f'\tbackground min, max = {scores_seg_bg.min()}, {scores_seg_bg.max()}')
    for mid_node_idx in range(num_nodes):
        graph_maxflow.add_tedge(nodes[mid_node_idx], scores_seg_fg[mid_node_idx], scores_seg_bg[mid_node_idx])

    graph_maxflow.maxflow()
    cut_segs_mask = graph_maxflow.get_grid_segments(nodes)
    cut_segs = mid_segments[cut_segs_mask]
    cut_segments_label = np.zeros_like(segments, dtype=np.uint8)
    for seg in cut_segs:
        cut_segments_label[segments==seg] = 1
    for seg in mid_segments[np.bitwise_not(cut_segs_mask)]:
        cut_segments_label[segments==seg] = 2

    result_fg_img = fg_mask.copy()
    for seg in mid_segments[np.bitwise_not(cut_segs_mask)]:
        result_fg_img[segments==seg] = True
    result_fg_img = skmorph.remove_small_objects(result_fg_img, 400)
    result_fg_img = skmorph.binary_closing(result_fg_img, skmorph.disk(5))

    cut_segments_img = skcolor.label2rgb(cut_segments_label, img, alpha=0.5)
    scores_fg_seg_img = set_segment_value(segments, mid_segments, scores_seg_fg)
    scores_bg_seg_img = set_segment_value(segments, mid_segments, scores_seg_bg)

    plt.subplot(241), plt.imshow(skseg.mark_boundaries(img[:,:,::-1], segments)), plt.imshow(fg_mask, cmap='cool', alpha=0.4), plt.imshow(bg_mask, cmap='Wistia', alpha=0.4)
    plt.subplot(242), plt.imshow(scores_fg_seg_img, cmap='gray'), plt.gca().set_title('foreground prob per segment', fontsize = FONT_SIZE)
    plt.subplot(243), plt.imshow(scores_bg_seg_img, cmap='gray'), plt.gca().set_title('background prob per segment', fontsize = FONT_SIZE)
    plt.subplot(244), plt.imshow(edge_map, cmap='gray'), plt.gca().set_title('edge map', fontsize = FONT_SIZE)
    plt.subplot(246), plt.imshow(cut_segments_img), plt.gca().set_title('graph-cut result', fontsize = FONT_SIZE)
    plt.subplot(247), plt.imshow(img[:,:,::-1]), plt.imshow(result_fg_img, cmap='gray', alpha=0.4), plt.gca().set_title('after healing', fontsize = FONT_SIZE)
    plt.savefig(f'{OUT_DIR}{file_name[0:-4]}.png', dpi = 2000)
