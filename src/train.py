import tensorflow as tf
import numpy as np
import os
from src.data_loader import load_data_paths, create_tf_dataset
from src.model import create_model, get_anchors
from src.box_utils import encode_boxes

BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-4 # Keep 1e-4 or lower for fine-tuning. Adam handles it well usually, but let's be safe.
# Actually for fine-tuning, usually 1e-4 is okay with Adam, but maybe 5e-5 is safer.
# Let's keep 1e-4 but unfreezing is the key.

def get_dataset(images_dir, annotations_dir, anchors, batch_size=32, split='train'):
    # Load paths
    data_list = load_data_paths(images_dir, annotations_dir)
    
    # Split
    np.random.seed(42)
    np.random.shuffle(data_list)
    split_idx = int(0.8 * len(data_list))
    
    if split == 'train':
        data_list = data_list[:split_idx]
    else:
        data_list = data_list[split_idx:]
        
    ds = create_tf_dataset(data_list, batch_size=batch_size, split=split)
    
    # Map to encode targets
    def encode_map_fn(images, targets):
        boxes, labels = targets
        # boxes: (batch, max_obj, 4) -> No, dataset yields unbatched first then batched? 
        # Wait, create_tf_dataset batches it.
        # But we want to encode per example usually, then batch.
        # Current create_tf_dataset yields batched data. 
        # But encode_boxes works on single image (N GT boxes).
        # We need to map unbatched.
        return images, targets

    # Re-create dataset unbatched to map encoding
    # Modify data_loader to return unbatched? 
    # Or just use map(encode) then batch.
    
    # Let's redefine create_tf_dataset behavior in data_loader or handle it here.
    # Actually, create_tf_dataset does batching.
    # Let's strip batching there or use unbatch() here.
    
    ds = ds.unbatch()
    
    def encoder(image, targets):
        gt_boxes = targets[0] # (max_obj, 4)
        gt_labels = targets[1] # (max_obj,)
        
        # Filter valid
        valid_mask = gt_labels >= 0
        valid_boxes = tf.boolean_mask(gt_boxes, valid_mask)
        valid_labels = tf.boolean_mask(gt_labels, valid_mask)
        
        target_boxes, target_labels = encode_boxes(valid_boxes, valid_labels, anchors)
        
        # target_labels is indices (M,). We need one-hot for Categorical Crossentropy?
        # Model outputs (batch, M, num_classes).
        # But we have 4 classes (3 + background).
        # Classes: 0,1,2. Background=3. Total 4.
        
        target_labels_onehot = tf.one_hot(target_labels, depth=4)
        
        return image, (target_labels_onehot, target_boxes)
        
    ds = ds.map(encoder, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

class FaceMaskLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.huber = tf.keras.losses.Huber()
        self.ce = tf.keras.losses.CategoricalCrossentropy(from_logits=False) # We have softmax in model

    def call(self, y_true, y_pred):
        # Unpack
        # y_true structure: dict or tuple?
        # Model returns [cls_output, reg_output]
        # In standardized fit, y_true matches y_pred structure if passed as list.
        # We'll pass y_true as (cls_target, reg_target) pair.
        pass

# Custom loss function wrappers
def classification_loss(y_true, y_pred):
    # y_true: (batch, M, 4) one-hot
    # y_pred: (batch, M, 3) ??? No, model outputs 3 classes.
    # We need model to output 4 classes to include background!
    # Correction: Model `num_classes` should be 4.
    
    # Or we use binary cross entropy for objectness?
    # Standard SSD: Background class is 0.
    # Let's update `model.py` to num_classes=4. (0,1,2 + 3=Background)
    
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def regression_loss(y_true, y_pred):
    # y_true: (batch, M, 4)
    # y_pred: (batch, M, 4)
    # We only want loss for positive anchors.
    # How do we know which are positive?
    # We can pass the label mask in y_true?
    # Tricky with standard Keras API.
    
    # Approach:
    # y_true_reg is (batch, M, 5) -> 4 coords + 1 mask?
    # Or rely on the fact that y_true_cls tells us which are background.
    
    # Let's write a custom Training Loop (train_step) or use a custom Layer for loss.
    # Custom training loop is often clearer for SSD.
    return tf.reduce_mean(tf.abs(y_true - y_pred)) # Place holder

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # y_true: (batch, M, 4) one-hot. Background is index 3.
        # y_pred: (batch, M, 4) softmax
        
        # Clip to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Calculate CE
        # - y_true * log(y_pred)
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weights
        # alpha * (1 - pt)^gamma for positive
        # (1 - alpha) * pt^gamma for negative ?? 
        # Actually standard definition: -alpha_t * (1 - pt)^gamma * log(pt)
        # where pt is prob of ground truth class.
        
        # We can implement simpler:
        weight = self.alpha * y_true * tf.pow((1 - y_pred), self.gamma)
        # For background (index 3), maybe we want less weight?
        # Usually alpha is for class balance. 
        # Let's trust standard focal loss formulation on all classes.
        
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

def main():
    # Setup
    data_dir = 'data'
    if os.path.exists(os.path.join(data_dir, 'face-mask-detection')):
        data_dir = os.path.join(data_dir, 'face-mask-detection')
        
    images_dir = os.path.join(data_dir, 'images')
    annotations_dir = os.path.join(data_dir, 'annotations')
    
    # 1. Update Model to 4 classes
    # 0: with_mask, 1: without, 2: incorrect, 3: background
    model = create_model(num_classes=4)
    
    # 2. Anchors
    anchors = get_anchors() # (M, 4)
    
    # 3. Dataset
    # Train heavily to debug
    train_ds = get_dataset(images_dir, annotations_dir, anchors, batch_size=BATCH_SIZE, split='train')
    
    # 4. Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # Loss
    focal_loss_fn = FocalLoss()
    
    # 5. Custom Training Loop
    @tf.function
    def train_step(images, targets):
        gt_cls, gt_reg = targets # gt_cls (B, M, 4), gt_reg (B, M, 4)
        
        with tf.GradientTape() as tape:
            pred_cls, pred_reg = model(images, training=True)
            
            # Classification Loss (Focal)
            # gt_cls is one-hot
            # Focal loss returns (B, M)
            c_loss_per_anchor = focal_loss_fn(gt_cls, pred_cls)
            c_loss = tf.reduce_mean(c_loss_per_anchor)
            
            # Regression Loss
            # Only for positive samples (label < 3)
            # gt_cls is one-hot. Background is index 3.
            # Positive mask: NOT background.
            background_mask = gt_cls[..., 3] # (B, M)
            positive_mask = 1.0 - background_mask
            
            # Localization loss (Smooth L1)
            loc_loss = tf.keras.losses.huber(gt_reg, pred_reg, delta=1.0) # (B, M)
            positive_count = tf.reduce_sum(positive_mask)
            loc_loss = tf.reduce_sum(loc_loss * positive_mask) / (positive_count + 1e-7)
            
            # optional debug: tf.print("Positives:", positive_count)
            
            total_loss = c_loss + loc_loss
            
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return total_loss, c_loss, loc_loss, positive_count

    # Loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        progbar = tf.keras.utils.Progbar(None)
        
        total_positives = 0
        for step, (images, targets) in enumerate(train_ds):
            loss, cl, ll, pos_count = train_step(images, targets)
            total_positives += pos_count
            
        print(f"Loss: {loss:.4f} Cls: {cl:.4f} Reg: {ll:.4f} Matches: {int(total_positives)}")
        
        # Validation?
        
    model.save('saved_models/face_mask_model.h5')

if __name__ == "__main__":
    main()
