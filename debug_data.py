import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.data_loader import load_data_paths, create_tf_dataset, CLASS_MAPPING
from src.box_utils import decode_boxes
from src.model import get_anchors
import os

# ID mapping
ID_TO_LABEL = {0: 'with_mask', 1: 'without_mask', 2: 'incorrect', 3: 'background'}

def visualize_batch(dataset, anchors):
    for images, targets in dataset.take(1):
        # targets: (cls_targets, reg_targets)
        # cls_targets: (B, M, 4)
        # reg_targets: (B, M, 4)
        
        cls_targets, reg_targets = targets
        
        # Take first image
        img = images[0]
        cls_t = cls_targets[0]
        reg_t = reg_targets[0]
        
        # Decode boxes to visualize "Ground Truth" as encoded
        # We need to find which anchors are positive
        # argmax of cls_t
        labels = tf.argmax(cls_t, axis=-1)
        
        # Filter positive
        # Background is 3
        pos_indices = tf.where(labels < 3)
        
        print(f"Number of positive anchors in this image: {len(pos_indices)}")
        
        if len(pos_indices) == 0:
            print("No positive anchors found! Check matching logic or anchor scales.")
            continue
            
        pos_anchors = tf.gather(anchors, pos_indices[:, 0])
        pos_reg = tf.gather(reg_t, pos_indices[:, 0])
        pos_labels = tf.gather(labels, pos_indices[:, 0])
        
        # Decode
        decoded_boxes = decode_boxes(pos_reg, pos_anchors)
        
        # Plot
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)
        h, w, _ = img.shape
        
        for box, lbl in zip(decoded_boxes, pos_labels):
            ymin, xmin, ymax, xmax = box.numpy()
            ymin, xmin, ymax, xmax = ymin*h, xmin*w, ymax*h, xmax*w
            
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin, ID_TO_LABEL[int(lbl)], color='g', weight='bold')
            
        plt.title("Visualizing Encoded Ground Truth (If empty, matching failed)")
        plt.show() # In non-interactive, this might not show, so we save.
        plt.savefig('debug_data_vis.png')
        print("Saved visualization to debug_data_vis.png")

if __name__ == "__main__":
    data_dir = 'data'
    if os.path.exists(os.path.join(data_dir, 'face-mask-detection')):
        data_dir = os.path.join(data_dir, 'face-mask-detection')
    images_dir = os.path.join(data_dir, 'images')
    annotations_dir = os.path.join(data_dir, 'annotations')
    
    from src.model import get_anchors
    anchors = get_anchors() # Using defaults from model.py
    
    # We need to recreate the dataset processing pipeline as in train.py
    # Copy-pasting logic from train.py get_dataset
    
    data_list = load_data_paths(images_dir, annotations_dir)
    np.random.shuffle(data_list)
    ds = create_tf_dataset(data_list[:10], batch_size=32, split='train')
    
    # We need the direct encoder logic since create_tf_dataset returns (img, (gt_boxes, gt_labels))
    # But train.py maps it to (img, (target_cls, target_reg))
    
    ds = ds.unbatch()
    
    from src.box_utils import encode_boxes
    
    def encoder(image, targets):
        gt_boxes = targets[0]
        gt_labels = targets[1]
        valid_mask = gt_labels >= 0
        valid_boxes = tf.boolean_mask(gt_boxes, valid_mask)
        valid_labels = tf.boolean_mask(gt_labels, valid_mask)
        
        target_boxes, target_labels = encode_boxes(valid_boxes, valid_labels, anchors)
        target_labels_onehot = tf.one_hot(target_labels, depth=4)
        return image, (target_labels_onehot, target_boxes)
        
    ds = ds.map(encoder)
    ds = ds.batch(1)
    
    visualize_batch(ds, anchors)
