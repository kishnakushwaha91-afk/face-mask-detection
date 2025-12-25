import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.box_utils import decode_boxes

COLORS = {0: 'g', 1: 'r', 2: 'y'}
LABELS = {0: 'with_mask', 1: 'without_mask', 2: 'mask_weared_incorrect'}

def process_predictions(cls_pred, reg_pred, anchors, conf_threshold=0.5, iou_threshold=0.5):
    """
    Apply NMS and filter predictions.
    cls_pred: (1, M, 4)
    reg_pred: (1, M, 4)
    """
    cls_pred = tf.squeeze(cls_pred, axis=0) # (M, 4)
    reg_pred = tf.squeeze(reg_pred, axis=0) # (M, 4)
    
    # Decode boxes
    pred_boxes = decode_boxes(reg_pred, anchors) # (M, 4) in [ymin, xmin, ymax, xmax]
    
    # Scores
    # Class 0,1,2 are valid. 3 is background.
    valid_probs = cls_pred[:, :3]
    scores = tf.reduce_max(valid_probs, axis=1)
    classes = tf.argmax(valid_probs, axis=1)
    
    # Filter
    mask = scores > conf_threshold
    filtered_boxes = tf.boolean_mask(pred_boxes, mask)
    filtered_scores = tf.boolean_mask(scores, mask)
    filtered_classes = tf.boolean_mask(classes, mask)
    
    if len(filtered_boxes) == 0:
        return [], [], []
        
    # NMS
    # tf.image.non_max_suppression expects [ymin, xmin, ymax, xmax]
    selected_indices = tf.image.non_max_suppression(
        filtered_boxes, filtered_scores, max_output_size=50, iou_threshold=iou_threshold
    )
    
    nms_boxes = tf.gather(filtered_boxes, selected_indices)
    nms_scores = tf.gather(filtered_scores, selected_indices)
    nms_classes = tf.gather(filtered_classes, selected_indices)
    
    return nms_boxes.numpy(), nms_classes.numpy(), nms_scores.numpy()

def visualize_detection(image, boxes, classes, scores):
    """
    Returns a matplotlib figure with detections.
    """
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    
    h, w, _ = image.shape
    
    for box, cls, score in zip(boxes, classes, scores):
        ymin, xmin, ymax, xmax = box
        # Scale
        ymin, xmin, ymax, xmax = ymin*h, xmin*w, ymax*h, xmax*w
        
        # Check bounds
        ymin = max(0, ymin)
        xmin = max(0, xmin)
        
        color = COLORS.get(cls, 'b')
        label_name = LABELS.get(cls, 'unknown')
        
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin-5, f'{label_name} {score:.2f}', color=color, fontsize=12, weight='bold', bbox=dict(facecolor='white', alpha=0.5))
        
    ax.axis('off')
    return fig
