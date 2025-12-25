import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np
import ssl
import os

# Fix for SSL certificate verify failed error on Mac
if os.environ.get('PYTHONHTTPSVERIFY', '') != '0':
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

def create_model(input_shape=(224, 224, 3), num_classes=3):
    """
    Creates a Face Mask Detection model based on MobileNetV2.
    """
    base_model = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Unfreeze base model for fine-tuning
    base_model.trainable = True
    
    # Feature map from MobileNetV2
    features = base_model.output
    
    # Anchor configuration
    # 5 aspect ratios * 4 scales = 20 anchors per cell
    num_anchors = 20 
    
    cls_head = layers.Conv2D(num_anchors * num_classes, (3, 3), padding='same', activation=None, name='class_head')(features)
    reg_head = layers.Conv2D(num_anchors * 4, (3, 3), padding='same', activation=None, name='reg_head')(features)
    
    # Reshape
    cls_output = layers.Reshape((-1, num_classes), name='class_reshape')(cls_head)
    reg_output = layers.Reshape((-1, 4), name='reg_reshape')(reg_head)
    
    cls_output = layers.Activation('softmax', name='class_final')(cls_output)
    
    model = models.Model(inputs=base_model.input, outputs=[cls_output, reg_output])
    return model

def get_anchors(feature_map_size=(7, 7), aspect_ratios=[1.0, 2.0, 0.5, 3.0, 0.33], scales=[0.1, 0.25, 0.5, 0.75]):
    """
    Generates anchor boxes.
    Returns: (total_anchors, 4) in [center_y, center_x, h, w]
    """
    grid_h, grid_w = feature_map_size
    anchors = []
    
    for i in range(grid_h):
        for j in range(grid_w):
            cy = (i + 0.5) / grid_h
            cx = (j + 0.5) / grid_w
            
            for ratio in aspect_ratios:
                for scale in scales:
                    # h/w = ratio (?) No, usually aspect ratio = w/h
                    # but let's stick to consistent definition.
                    # if ratio = w/h
                    # w = h * ratio
                    # w*h = scale^2 (area)
                    # h^2 * ratio = scale^2 => h = scale / sqrt(ratio)
                    # w = scale * sqrt(ratio)
                    
                    h = scale / np.sqrt(ratio)
                    w = scale * np.sqrt(ratio)
                    
                    anchors.append([cy, cx, h, w])
                    
    return np.array(anchors, dtype=np.float32)
