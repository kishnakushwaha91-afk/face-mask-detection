# Face Mask Detection System

A MobileNetV2-based SSD implementation for detecting face masks.

## Setup

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Dataset**
    Downloads from Kaggle (Requires `kaggle.json` in `~/.kaggle/` or environment variables).
    ```bash
    python download_data.py
    ```
    *Alternatively, place `images` and `annotations` folders inside `data/`.*

## Usage

### Training
Train the model (adjust batch size/epochs in `src/train.py` if needed):
```bash
python -m src.train
```

### Evaluation
Evaluate on test images and visualize:
```bash
python -m src.evaluate
```

### Web App
Run the Streamlit application:
```bash
streamlit run app.py
```

## Structure
- `src/model.py`: SSD-lite architecture.
- `src/train.py`: Custom training loop with regression/classification loss.
## Deployment on Streamlit Cloud

1.  **Fork/Clone this repository**.
2.  **Dataset & Model**:
    *   **Option A (Recommended)**: Upload your `saved_models/face_mask_model.h5` to the GitHub repository (if < 100MB) or use Git LFS.
    *   **Option B**: Train on cloud. (Not recommended for free tier).
3.  **Setup on Streamlit Cloud**:
    *   Connect your GitHub repository.
    *   Set the main file to `app.py`.
    *   The app will automatically install dependencies from `requirements.txt`.
