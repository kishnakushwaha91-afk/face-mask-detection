import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.data_loader import load_data_paths, create_tf_dataset, CLASS_MAPPING
from src.model import create_model, get_anchors
from src.box_utils import decode_boxes
from src.inference_utils import process_predictions, visualize_detection

# Reverse mapping
ID_TO_LABEL = {v: k for k, v in CLASS_MAPPING.items()}
COLORS = {'with_mask': 'g', 'without_mask': 'r', 'mask_weared_incorrect': 'y'}

def load_trained_model(model_path, num_classes=4):
    model = create_model(num_classes=num_classes)
    model.load_weights(model_path)
    return model


def compute_map(model, test_ds, anchors):
    # Simplified mAP placeholder
    # Real mAP requires accumulating all predictions and GTs, sorting by score, matching, etc.
    print("Computing mAP (Placeholder)...")
    return 0.0

def main():
    model_path = 'saved_models/face_mask_model.h5'
    if not os.path.exists(model_path):
        print("Model not found. Train first.")
        return

    model = load_trained_model(model_path)
    anchors = get_anchors()

    # Load a few test images
    data_dir = 'data'
    if os.path.exists(os.path.join(data_dir, 'face-mask-detection')):
        data_dir = os.path.join(data_dir, 'face-mask-detection')
        
    images_dir = os.path.join(data_dir, 'images')
    # Just list some files
    import glob
    test_images = sorted(glob.glob(os.path.join(images_dir, '*.png')))[:10]
    
    for img_path in test_images:
        original_img = tf.io.read_file(img_path)
        original_img = tf.image.decode_image(original_img, channels=3)
        original_h, original_w = original_img.shape[:2]
        
        # Preprocess
        img = tf.image.resize(original_img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        img_batch = tf.expand_dims(img, axis=0)
        
        cls_pred, reg_pred = model.predict(img_batch)
        
        boxes, classes, scores = process_predictions(cls_pred[0], reg_pred[0], anchors)
        
        print(f"Image: {os.path.basename(img_path)}, Detections: {len(boxes)}")
        if len(boxes) > 0:
            visualize_detection(img.numpy(), boxes, classes, scores, save_path=f"output_{os.path.basename(img_path)}")

if __name__ == "__main__":
    main()
