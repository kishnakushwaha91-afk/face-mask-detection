import opendatasets as od
import os

dataset_url = 'https://www.kaggle.com/datasets/andrewmvd/face-mask-detection'
data_dir = 'data'

if __name__ == "__main__":
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    print("Downloading dataset...")
    od.download(dataset_url, data_dir)
    
    # Move files if nested
    # The dataset usually comes as data/face-mask-detection/...
    # We want data/images and data/annotations
    
    base_path = os.path.join(data_dir, 'face-mask-detection')
    if os.path.exists(base_path):
        print("Organizing dataset...")
        # Start by assuming standard structure
        # If downloaded, check structure
        pass
