import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.segmentation as skseg
import skimage.filters as skfilter
import skimage.io   as skio
from skimage.future import graph
import skimage.color as skcolor

import maxflow

import prob_map
import util

FG_NODE = 1
BG_NODE = 2
BG_BDR_FLAG = 3
MID_NODE = 5
POSSIBLE_BDR_FLAG = 6

def superpixel_graph(img, fg_mask, bg_mask, mid_mask):
    #masked_img = cv.bitwise_and(img, img, mask=mid_mask.astype(np.uint8))
    masked_img = img

    img_1 = img[:,:,::-1]
    img_1 = skfilter.gaussian(img_1, 5)
    edges = skfilter.sobel(skcolor.rgb2gray(img_1))

    segments = skseg.slic(masked_img, n_segments=5000, convert2lab=True)
    unq_segments = np.unique(segments)
    n_segments = len(unq_segments)

    g = graph.rag_boundary(segments, edges)

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

DIR = "D:\Projects\Oh\data\images\silhouette\images\\view_178"
PROB_MAP_DIR = 'D:\Projects\Oh\data\images\silhouette\\avg_silhouette_map'
img_label_mapping = 'D:\Projects\Oh\data\images\silhouette\mapping_image_avg_silhouette.txt'


for file_name in os.listdir(DIR):
    if file_name not in 'image178_laxsquadT_lBlue_LL_0427.jpg':
        continue

    img = util.preprocess_img(cv.imread(f'{DIR}/{file_name}'))

    prb_label = prob_map.find_mean_silhouette_label(img_label_mapping, file_name)
    prb_img = prob_map.load_mean_silhouette(PROB_MAP_DIR, prb_label)
    fg_mask, bg_mask, mask = prob_map.build_fg_bg_masks(prb_img, False, fg_threshold = 0.95, as_bool = True)
    mid_mask = np.bitwise_and(np.bitwise_not(fg_mask), np.bitwise_not(bg_mask))

    graph_segments, node_marks, segments = superpixel_graph(img, fg_mask, bg_mask, mid_mask)
    mid_segments_mask = (node_marks==MID_NODE)
    mid_segments_mapping = np.cumsum(mid_segments_mask.astype(np.uint32))
    mid_segments = np.where(mid_segments_mask==True)[0]
    mid_mean_colour = np.array([np.mean(img[segments == idx], axis=0) for idx in mid_segments])

    # labels = np.zeros_like(segments, dtype=np.uint8)
    # for node_idx in g:
    #     if node_marks[node_idx] == FG_FLAG:
    #         labels[segments==node_idx] = 1
    #     elif node_marks[node_idx] == BG_FLAG:
    #         labels[segments==node_idx] = 2
    #     else:
    #         labels[segments==node_idx] = 3
    # lb_color = skcolor.label2rgb(labels, img)
    # plt.imshow(skseg.mark_boundaries(lb_color, segments))
    # plt.show()

    gmm_fg = prob_map.estimate_gmm(img, fg_mask, 9, 'full')
    gmm_bg = prob_map.estimate_gmm(img, bg_mask, 9, 'full')

    scores_fg = gmm_fg.score_samples(mid_mean_colour)
    scores_bg = gmm_bg.score_samples(mid_mean_colour)

    # plt.subplot(121); plt.imshow(fg_prob_map)
    # plt.subplot(122); plt.imshow(bg_prob_map)
    # plt.show()

    num_nodes = np.prod(mid_segments.shape)
    graph_maxflow = maxflow.Graph[float](num_nodes, num_nodes*4)
    nodes = graph_maxflow.add_nodes(num_nodes)

    for x, y, d in graph_segments.edges_iter(data=True):
        if mid_segments_mask[x] == True and mid_segments_mask[y] == True:
            mid_node_idx_x = nodes[mid_segments_mapping[x]-1]
            mid_node_idx_y = nodes[mid_segments_mapping[y]-1]
            smoothess_cost = d['weight']
            graph_maxflow.add_edge(nodes[mid_node_idx_x], nodes[mid_node_idx_y], smoothess_cost, smoothess_cost)

    for mid_node_idx in range(num_nodes):
        graph_maxflow.add_tedge(nodes[mid_node_idx], scores_fg[mid_node_idx], scores_bg[mid_node_idx])

    graph_maxflow.maxflow()
    cut_segs = graph_maxflow.get_grid_segments(nodes)
    print(cut_segs)