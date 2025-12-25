import tensorflow as tf
import numpy as np

def compute_iou(boxes1, boxes2):
    """
    Computes pairwise IOU matrix.
    boxes1: (N, 4) [ymin, xmin, ymax, xmax]
    boxes2: (M, 4) [ymin, xmin, ymax, xmax] (Anchors converted to corners)
    Returns: (N, M)
    """
    # Expand dims for broadcasting
    # boxes1: (N, 1, 4)
    # boxes2: (1, M, 4)
    b1 = tf.expand_dims(boxes1, axis=1)
    b2 = tf.expand_dims(boxes2, axis=0)
    
    # Intersections
    ymin = tf.maximum(b1[..., 0], b2[..., 0])
    xmin = tf.maximum(b1[..., 1], b2[..., 1])
    ymax = tf.minimum(b1[..., 2], b2[..., 2])
    xmax = tf.minimum(b1[..., 3], b2[..., 3])
    
    h = tf.maximum(0.0, ymax - ymin)
    w = tf.maximum(0.0, xmax - xmin)
    intersection = h * w
    
    # Unions
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union = area1 + area2 - intersection
    
    iou = intersection / (union + 1e-7)
    return iou

def convert_anchors_to_corners(anchors):
    """
    anchors: [cy, cx, h, w]
    returns: [ymin, xmin, ymax, xmax]
    """
    cy, cx, h, w = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    ymin = cy - h / 2.0
    xmin = cx - w / 2.0
    ymax = cy + h / 2.0
    xmax = cx + w / 2.0
    return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

def encode_boxes(gt_boxes, gt_labels, anchors, match_threshold=0.5):
    """
    Encodes ground truth boxes to anchor targets.
    gt_boxes: (N, 4) [ymin, xmin, ymax, xmax]
    gt_labels: (N,) class indices
    anchors: (M, 4) [cy, cx, h, w]
    
    Returns:
    target_boxes: (M, 4) encoded deltas
    target_labels: (M, num_classes) or (M,) 
                   We will return one-hot or index. Let's return index.
                   0: With Mask, 1: Without Mask, 2: Incorrect, 3: Background
                   So we shift user classes +1 ? 
                   Actually, let's keep 0-2 as classes and use 'background' implicitly or explicitly.
                   Dataset has 3 classes. So Background=3?
                   Or Background=0 and shift others?
                   Let's say Background is the Last Class => 3.
    """
    # Convert anchors to corners for IOU
    anchor_corners = convert_anchors_to_corners(anchors)
    
    # Calc IOU
    iou_mat = compute_iou(gt_boxes, anchor_corners) # (N_gt, M_anchors)
    
    # For each anchor, find max IOU with any GT
    anchor_max_iou = tf.reduce_max(iou_mat, axis=0)
    anchor_max_idx = tf.argmax(iou_mat, axis=0)
    
    # Assign labels
    # Background = 3
    targets_labels = tf.fill((len(anchors),), 3) 
    
    # Matches
    matched_mask = anchor_max_iou > match_threshold
    
    matched_gt_idx = tf.boolean_mask(anchor_max_idx, matched_mask)
    matched_labels = tf.gather(gt_labels, matched_gt_idx)
    
    # Iterate to assign (using tensor_scatter_nd_update is better in graph, but numpy is easier eager)
    # Since this runs in tf.data pipeline, we need TF ops.
    
    indices = tf.where(matched_mask)
    targets_labels = tf.tensor_scatter_nd_update(targets_labels, indices, matched_labels)
    
    # Encode boxes for matched only (others can be 0 or ignored in loss)
    # Delta encoding
    # tx = (gx - ax) / aw
    # ty = (gy - ay) / ah
    # tw = log(gw / aw)
    # th = log(gh / ah)
    
    matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
    matched_anchors = tf.gather(anchors, indices[:, 0])
    
    # Convert matched GT from corners to center/size
    g_ymin, g_xmin, g_ymax, g_xmax = tf.unstack(matched_gt_boxes, axis=1)
    g_h = g_ymax - g_ymin
    g_w = g_xmax - g_xmin
    g_cy = g_ymin + g_h * 0.5
    g_cx = g_xmin + g_w * 0.5
    
    a_cy, a_cx, a_h, a_w = tf.unstack(matched_anchors, axis=1)
    
    tx = (g_cx - a_cx) / a_w
    ty = (g_cy - a_cy) / a_h
    tw = tf.math.log(g_w / a_w + 1e-7)
    th = tf.math.log(g_h / a_h + 1e-7)
    
    encoded_deltas = tf.stack([ty, tx, th, tw], axis=1) # TF order: y, x, h, w ? Or x, y, w, h?
    # model output reg_head predicts 4 values.
    # Let's align with our encode: [ty, tx, th, tw]
    
    target_boxes = tf.zeros_like(anchors)
    target_boxes = tf.tensor_scatter_nd_update(target_boxes, indices, encoded_deltas)
    
    return target_boxes, targets_labels

def decode_boxes(encoded_deltas, anchors):
    """
    Decodes deltas back to corners.
    encoded_deltas: [ty, tx, th, tw]
    anchors: [cy, cx, h, w]
    """
    ty, tx, th, tw = tf.unstack(encoded_deltas, axis=-1)
    a_cy, a_cx, a_h, a_w = tf.unstack(anchors, axis=-1)
    
    pre_cy = ty * a_h + a_cy
    pre_cx = tx * a_w + a_cx
    pre_h = tf.exp(th) * a_h
    pre_w = tf.exp(tw) * a_w
    
    ymin = pre_cy - pre_h / 2.0
    xmin = pre_cx - pre_w / 2.0
    ymax = pre_cy + pre_h / 2.0
    xmax = pre_cx + pre_w / 2.0
    
    return tf.stack([ymin, xmin, ymax, xmax], axis=-1)
