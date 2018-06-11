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
exp_type = 'bg_sub_'

CUR_FILENAME = ''

def superpixel_graph(img, fg_mask, bg_mask, edge_map):
    masked_img = cv.bitwise_and(img, img, mask=mid_mask.astype(np.uint8))

    segments = skseg.slic(masked_img, n_segments=50000, convert2lab=True)
    unq_segments = np.unique(segments)
    n_segments = len(unq_segments)

    #create a graph whose nodes are superpixel segment
    #edges of this graph store edge strength difference between neighbor superpixel segments
    g = graph.rag_boundary(segments, edge_map)

    # lc = graph.show_rag(segments, g, edges, img_cmap='gray', edge_cmap='viridis', edge_width=1.2)
    # plt.colorbar(lc, fraction=0.03)
    # skio.show()

    fg_px_cnt   = np.zeros((n_segments,), dtype=np.uint32)
    bg_px_cnt   = np.zeros((n_segments,), dtype=np.uint32)
    total_cnt   = np.zeros((n_segments,), dtype=np.uint32)
    node_marks  = np.zeros((n_segments,), dtype=np.uint32)

    #count number of foreground pixel, backgroud per segment
    for index in np.ndindex(segments.shape):
        current = segments[index]
        fg_px_cnt[current] += int(fg_mask[index])
        bg_px_cnt[current] += int(bg_mask[index])
        total_cnt[current] += 1

    #classify segments into sure fg, bg and mid types
    for node_idx in g:
        fg_percent = float(fg_px_cnt[node_idx])/ float(total_cnt[node_idx])
        bg_percent = float(bg_px_cnt[node_idx])/ float(total_cnt[node_idx])
        #possible_cnt = total_cnt[node_idx] - bg_px_cnt[node_idx] - fg_px_cnt[node_idx]
        if bg_percent > 0.9:
            node_marks[node_idx] = BG_NODE
        elif fg_percent > 0.9:
            node_marks[node_idx] = FG_NODE
        else:
            node_marks[node_idx] = MID_NODE

    return g, node_marks, segments

def rescale_img(img, indices, out_range=(0.0,1.0)):
    minval = img[indices[0], indices[1]].min()
    maxval = img[indices[0], indices[1]].max()
    img[mid_rows, mid_cols] = out_range[0] + (out_range[1] - out_range[0]) * (img[mid_rows, mid_cols] - minval)/(maxval-minval)
    return img

def set_segment_value(all_segments, segments, values, out_range = (0,1)):
    value_img = np.zeros_like(all_segments, dtype=np.float32)
    values = out_range[0] + (out_range[1]-out_range[0]) * (values - values.min()) / (values.max() - values.min())
    for i, seg in enumerate(segments):
        value_img[all_segments==seg] = values[i]
    return value_img

def calc_prob_images(img, fg_mask, bg_mask, mid_mask_not, mid_mask_idxs):
    bg_mask_dilated = skmorph.binary_dilation(np.bitwise_not(bg_mask), skmorph.rectangle(50,50))
    bg_train_sample_mask = np.bitwise_and(bg_mask_dilated, bg_mask)
    mid_samples = img[mid_mask_idxs[0], mid_mask_idxs[1]]

    fg_samples_mask = np.bitwise_and(np.bitwise_not(skmorph.erosion(fg_mask, skmorph.disk(25))), fg_mask)
    scores_fg_img = np.zeros_like(fg_mask, dtype=np.float32)
    gmm_fg = prob_map.estimate_gmm(img, fg_samples_mask, 5, 'full')
    scores_fg_img[mid_mask_idxs[0], mid_mask_idxs[1]] = gmm_fg.score_samples(mid_samples)
    print(f'\tforeground min, max = {scores_fg_img.min()}, {scores_fg_img.max()}')
    scores_fg_img = np.exp(scores_fg_img)
    scores_fg_img = skexposure.adjust_gamma(scores_fg_img, 0.2)
    #scores_fg_img = ndi.median_filter(scores_fg_img, size=2)
    #scores_fg_img = rescale_img(scores_fg_img, mid_mask_idxs, (-200, 0))
    scores_fg_img[mid_mask_not] = 0

    scores_bg_img = np.zeros_like(fg_mask, dtype=np.float32)
    gmm_bg = prob_map.estimate_gmm(img, bg_train_sample_mask, 5, 'full')
    scores_bg_img[mid_mask_idxs[0], mid_mask_idxs[1]] = gmm_bg.score_samples(mid_samples)
    print(f'\tbackground min, max = {scores_bg_img.min()}, {scores_bg_img.max()}')
    scores_bg_img = np.exp(scores_bg_img)
    scores_bg_img = skexposure.adjust_gamma(scores_bg_img, 0.2)
    #scores_bg_img = rescale_img(scores_bg_img, mid_mask_idxs, (-200, 0))
    #scores_bg_img = ndi.median_filter(scores_bg_img, size=2)
    scores_bg_img[mid_mask_not] = 0

    plt.clf()
    plt.subplot(131), plt.imshow(img); plt.imshow(fg_samples_mask, alpha=0.4)
    plt.subplot(132), plt.imshow(scores_fg_img, cmap='gray')
    plt.subplot(133), plt.imshow(scores_bg_img, cmap='gray')
    plt.savefig(f'{OUT_DIR_PROB_MAP}{CUR_FILENAME}.png', dpi = 1000)

    return scores_fg_img, scores_bg_img

def calc_prob_map_superpixel_segment(scores_fg_img, scores_bg_img, segments, mid_segments, mid_mask_idxs):
    scores_seg_fg = np.array([np.mean(scores_fg_img[segments == idx], axis=0) for idx in mid_segments])
    scores_seg_bg = np.array([np.mean(scores_bg_img[segments == idx], axis=0) for idx in mid_segments])

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
    edge_map = skfilter.sobel(skfilter.gaussian(skcolor.rgb2gray(img_1), sigma=5), mask= mask)
    #edge_map = skfilter.sobel(skcolor.rgb2gray(img_1), mask= mask)
    edge_map = skexposure.equalize_adapthist(edge_map, kernel_size=30)
    edge_map[np.bitwise_not(mask)] = 0
    return edge_map

def suplement_edge_map_with_prob_maps(edge_map, mid_mask, fg_prob_map, bg_prob_map):
    fg_edge_map = skfilter.sobel(skfilter.gaussian(fg_prob_map, sigma=5), mask = mid_mask)
    bg_edge_map = skfilter.sobel(skfilter.gaussian(bg_prob_map, sigma=5), mask = mid_mask)

    blended_edge_map = 0.4 *edge_map + 0.3 * fg_edge_map + 0.3 * bg_edge_map
    plt.clf()
    plt.subplot(231); plt.imshow(edge_map),    plt.gca().set_title('sobel edge from input img + increase local contrast', fontsize= FONT_SIZE)
    plt.subplot(232); plt.imshow(fg_edge_map), plt.gca().set_title('sobel edge from foreground prob map', fontsize= FONT_SIZE)
    plt.subplot(233); plt.imshow(bg_edge_map), plt.gca().set_title('sobel edge from background prob map', fontsize= FONT_SIZE)
    plt.subplot(235); plt.imshow(blended_edge_map), plt.gca().set_title('blended edge: 0.4*e0 + 0.3*e1 + 0.3*e2', fontsize= FONT_SIZE)
    plt.savefig(f'{OUT_DIR_EDGE_MAP}{CUR_FILENAME}.png', dpi=1000)

    return blended_edge_map

plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

for file_name in os.listdir(DIR):
    #if file_name not in 'image178_laxsquadT_mBBlue_LL_0427.jpg':
    #    continue
    CUR_FILENAME = file_name[0:-4]

    print(f'processing file {file_name}')
    img = util.preprocess_img(cv.imread(f'{DIR}{file_name}'))
    img = util.remove_light_tubes(img, black=True)

    print('\tload mean silhouette')
    prb_label = prob_map.find_mean_silhouette_label(img_label_mapping, file_name)
    prb_img = prob_map.load_mean_silhouette(PROB_MAP_DIR, prb_label)

    print('\tcalculate sure foreground, sure background and possible masks from mean silhouette')
    fg_mask, bg_mask, mask = prob_map.build_fg_bg_masks(prb_img, False, fg_threshold = 0.98, as_bool = True)
    mid_mask = np.bitwise_and(np.bitwise_not(fg_mask), np.bitwise_not(bg_mask))
    mid_mask_not = np.bitwise_not(mid_mask)
    mid_rows, mid_cols = np.where(mid_mask==True)
    mid_mask_idxs = (mid_rows, mid_cols)

    print('\testimate probability map')
    scores_fg_img, scores_bg_img = calc_prob_images(img, fg_mask, bg_mask, mid_mask_not, mid_mask_idxs)

    print('\tcalculate edge map')
    edge_map = calc_edge_map(img, mid_mask)
    edge_map = suplement_edge_map_with_prob_maps(edge_map, mid_mask,
                                                 rescale_img(scores_fg_img, mid_mask_idxs),
                                                 rescale_img(scores_bg_img, mid_mask_idxs))

    #rescale probabilities back to its standard range.
    scores_fg_img = rescale_img(scores_fg_img, mid_mask_idxs, (-200, 0))
    scores_bg_img = rescale_img(scores_bg_img, mid_mask_idxs, (-200, 0))

    print('\tsuperpixel graph')
    graph_segments, node_marks, segments = superpixel_graph(img, fg_mask, bg_mask, edge_map)
    mid_segments_mask = (node_marks==MID_NODE)
    mid_segments_mapping = np.cumsum(mid_segments_mask.astype(np.uint32))
    mid_segments = np.where(mid_segments_mask==True)[0]

    #calculate fg and bg probabilities for superpixel segments
    scores_seg_fg, scores_seg_bg = calc_prob_map_superpixel_segment(scores_fg_img, scores_bg_img, segments, mid_segments, mid_mask_idxs)

    print('\tbuild the max-flow graph')
    num_nodes = np.prod(mid_segments.shape)
    graph_maxflow = maxflow.Graph[float](num_nodes, num_nodes*4)
    nodes = graph_maxflow.add_nodes(num_nodes)

    print('\t set weigts for edges of max-flow graph')

    #1. smoothness costs for edges between neighbor nodes
    #these weights are taken from the superpixel graph we generated above
    for x, y, d in graph_segments.edges_iter(data=True):
        if mid_segments_mask[x] == True and mid_segments_mask[y] == True:
            mid_node_idx_x = nodes[mid_segments_mapping[x]-1]
            mid_node_idx_y = nodes[mid_segments_mapping[y]-1]
            smoothess_cost = d['weight']
            graph_maxflow.add_edge(nodes[mid_node_idx_x], nodes[mid_node_idx_y], smoothess_cost, smoothess_cost)

    #2. fg and bg probabilities per node
    for mid_node_idx in range(num_nodes):
        graph_maxflow.add_tedge(nodes[mid_node_idx], scores_seg_fg[mid_node_idx], scores_seg_bg[mid_node_idx])

    #do a cut
    graph_maxflow.maxflow()

    #label the nodes after from the cut
    cut_segs_mask = graph_maxflow.get_grid_segments(nodes)
    cut_segs = mid_segments[cut_segs_mask]
    cut_segments_label = np.zeros_like(segments, dtype=np.uint8)
    for seg in cut_segs:
        cut_segments_label[segments==seg] = 1
    for seg in mid_segments[np.bitwise_not(cut_segs_mask)]:
        cut_segments_label[segments==seg] = 2

    #colorize segments based on their labels
    cut_segments_img = skcolor.label2rgb(cut_segments_label, img, alpha=0.5)

    #build the illustration outputs
    result_fg_img = fg_mask.copy()
    for seg in mid_segments[np.bitwise_not(cut_segs_mask)]:
        result_fg_img[segments==seg] = True

    result_fg_img = skmorph.remove_small_objects(result_fg_img, 1000)
    result_fg_img = skmorph.binary_closing(result_fg_img, skmorph.disk(10))

    scores_fg_seg_img = set_segment_value(segments, mid_segments, scores_seg_fg)
    scores_bg_seg_img = set_segment_value(segments, mid_segments, scores_seg_bg)

    plt.subplot(241), plt.imshow(skseg.mark_boundaries(img[:,:,::-1], segments)), plt.imshow(fg_mask, cmap='cool', alpha=0.4), plt.imshow(bg_mask, cmap='Wistia', alpha=0.4)
    plt.subplot(242), plt.imshow(scores_fg_seg_img, cmap='gray'), plt.gca().set_title('foreground prob per segment', fontsize = FONT_SIZE)
    plt.subplot(243), plt.imshow(scores_bg_seg_img, cmap='gray'), plt.gca().set_title('background prob per segment', fontsize = FONT_SIZE)
    plt.subplot(244), plt.imshow(edge_map, cmap='gray'), plt.gca().set_title('edge map', fontsize = FONT_SIZE)
    plt.subplot(246), plt.imshow(cut_segments_img), plt.gca().set_title('graph-cut result', fontsize = FONT_SIZE)
    plt.subplot(247), plt.imshow(img[:,:,::-1]), plt.imshow(result_fg_img, cmap='gray', alpha=0.4), plt.gca().set_title('after healing', fontsize = FONT_SIZE)
    plt.savefig(f'{OUT_DIR}{file_name[0:-4]}.png', dpi = 2000)
