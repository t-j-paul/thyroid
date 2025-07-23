# Thyroid Nodule Classifier

An AI-powered diagnostic tool for classifying thyroid nodules (benign vs. malignant) from ultrasound images using a ResNet18 CNN and providing interpretive Grad-CAM visualizations via an interactive Streamlit dashboard.

## Project Structure

```
thyroid_nodule_classifier/
├── data/
│   ├── raw/                  # Raw ultrasound cine-clips (video files)
│   └── processed/            # Processed image frames
├── notebooks/                # Jupyter notebooks for EDA and prototyping
├── src/
│   ├── preprocessing/
│   │   ├── extract_frames.py
│   │   └── clean_metadata.py
│   ├── models/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── inference.py
│   └── utils/
│       ├── visualization.py
│       └── logging.py
├── app/
│   └── streamlit_app.py
├── tests/
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

##  Quick Start

### 1. Environment Setup

- Requires **Python 3.10+**
- Create a virtual environment:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- Install dependencies:
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

### 2. Data Preprocessing

- **Extract frames from cine-clips:**
  ```bash
  python src/preprocessing/extract_frames.py --input_dir data/raw/ --output_dir data/processed/
  ```
- **Clean and integrate metadata:**
  ```bash
  python src/preprocessing/clean_metadata.py --metadata_csv data/raw/metadata.csv --output_csv data/processed/clean_metadata.csv
  ```

### 3. Model Training

- **Train the ResNet18 classifier:**
  ```bash
  python src/models/train.py --config configs/train_config.yaml
  ```
  - Model checkpoints are saved with SHA-256 hashes for verification.

### 4. Evaluation

- **Evaluate model performance:**
  ```bash
  python src/models/evaluate.py --checkpoint_path checkpoints/best_model.pth
  ```
  - Generates metrics and plots (ROC, confusion matrix, etc.).

### 5. Inference & Visualization

- **Run inference with model integrity check and Grad-CAM:**
  ```bash
  python src/models/inference.py --image_path path/to/image.png --checkpoint_path checkpoints/best_model.pth
  ```

### 6. Streamlit Dashboard

- **Start local dashboard:**
  ```bash
  streamlit run app/streamlit_app.py
  ```
- **Features:**
  - Upload ultrasound frame(s), get malignancy predictions and Grad-CAM explanations.
  - Real-time analytics: ROC, confusion matrix, class distributions.
  - Secure file handling, session isolation, comprehensive logging.

### 7. Docker & Azure Deployment

- **Build Docker image:**
  ```bash
  docker build -t thyroid-nodule-classifier .
  ```
- **Run container locally:**
  ```bash
  docker run -p 8501:8501 thyroid-nodule-classifier
  ```
- **Deploy to Azure (App Service, Container Instance, or Azure ML):**
  - Ensure HTTPS is enabled for all endpoints.
  - Refer to Azure deployment docs for Streamlit.

### 8. Testing

- **Run unit/integration tests:**
  ```bash
  pytest tests/
  ```

---

## 🛡 Security & Maintenance

- **Model file integrity:** All model checkpoints are verified via SHA-256 hash before inference.
- **Secure connections:** Enforce HTTPS for dashboard deployment.
- **File upload security:** Only allow image files, enforce size/type limits.
- **Logging:** All user actions, predictions, and model metrics are logged (see `src/utils/logging.py`).
- **Drift monitoring:** Model accuracy is monitored and logged over time.

---

## Example Commands

- Run a sample prediction:
  ```bash
  streamlit run app/streamlit_app.py
  ```
  - Upload a frame from `data/processed/` to test.

- View logs:
  ```
  tail -f logs/activity.log
  ```

---

##  License & Citation

*Add your license/citation here.*
```
