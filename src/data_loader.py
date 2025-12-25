import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
from PIL import Image

CLASS_MAPPING = {
    'with_mask': 0,
    'without_mask': 1,
    'mask_weared_incorrect': 2
}

def parse_annotation(xml_file):
    """
    Parses XML file and returns boxes (absolute) and labels.
    Attempts to read image size from XML, returns None for size if missing.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size_node = root.find('size')
    width = int(size_node.find('width').text) if size_node is not None else None
    height = int(size_node.find('height').text) if size_node is not None else None
    
    boxes = []
    labels = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_MAPPING:
            continue
            
        label = CLASS_MAPPING[name]
        
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
        
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int32), (width, height)

def normalize_boxes(boxes, width, height):
    """
    Normalizes boxes to [0, 1] range.
    Converts from [xmin, ymin, xmax, ymax] to [ymin, xmin, ymax, xmax] (TF convention).
    """
    # [xmin, ymin, xmax, ymax]
    boxes_norm = boxes.copy()
    boxes_norm[:, 0] = boxes[:, 1] / height # ymin
    boxes_norm[:, 1] = boxes[:, 0] / width  # xmin
    boxes_norm[:, 2] = boxes[:, 3] / height # ymax
    boxes_norm[:, 3] = boxes[:, 2] / width  # xmax
    return boxes_norm

def preprocess_image(image_path, target_size):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def load_data_paths(images_dir, annotations_dir):
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    data = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        xml_file = os.path.join(annotations_dir, base_name + '.xml')
        if os.path.exists(xml_file):
            data.append({'image': os.path.join(images_dir, img_file), 'xml': xml_file})
    return data

def create_tf_dataset(data_list, batch_size=32, target_size=(224, 224), split='train'):
    """
    Creates a tf.data.Dataset.
    """
    max_objects = 100
    
    def generator():
        for item in data_list:
            boxes, labels, original_size = parse_annotation(item['xml'])
            if len(boxes) == 0:
                continue
                
            # Read size if not in XML
            if original_size[0] is None:
                with Image.open(item['image']) as img:
                    width, height = img.size
            else:
                width, height = original_size
                
            norm_boxes = normalize_boxes(boxes, width, height)
            
            img_tensor = preprocess_image(item['image'], target_size)
            
            # Pad
            padded_boxes = np.zeros((max_objects, 4), dtype=np.float32)
            padded_labels = np.full((max_objects,), -1, dtype=np.int32)
            
            num = len(boxes)
            padded_boxes[:min(num, max_objects)] = norm_boxes[:max_objects]
            padded_labels[:min(num, max_objects)] = labels[:max_objects]
            
            yield img_tensor, (padded_boxes, padded_labels)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(target_size[0], target_size[1], 3), dtype=tf.float32),
            (
                tf.TensorSpec(shape=(max_objects, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(max_objects,), dtype=tf.int32)
            )
        )
    )
    
    # Shuffle and batch
    if split == 'train':
        dataset = dataset.shuffle(buffer_size=500)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
