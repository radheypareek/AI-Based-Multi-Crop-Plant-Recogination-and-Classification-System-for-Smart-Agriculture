# Submission Package – Multi-Crop Plant Recognition

This folder contains everything needed to reproduce training and run inference/web app.

Dataset Link :- https://www.kaggle.com/datasets/omrathod2003/140-most-popular-crops-image-dataset

## Contents
- `submission_manifest.txt` – authoritative list of files/folders to include
- `REPORT.md` – project report with topic-wise tables
- Top-level project files copied here at packaging time per manifest

## At-a-Glance
- Best Validation Accuracy (Top‑1): [PENDING]
- Classes: 154
- Model: EfficientNet‑B0 or MobileNetV2 (speed profile)
- Image size: 160–192 px
- Inference: `src/infer.py` (+ Streamlit app with Gemini)

## Quickstart

1) Create environment
- pip: `pip install -r requirements.txt`
- conda (optional): `conda env create -f environment.yml`

2) Prepare dataset (if not already processed)
- Place raw images per class under `data/raw/<class_name>/*.jpg`
- Run: `python src/prepare_dataset.py --input data/raw --output data/RGB_224x224 --val_split 0.15 --test_split 0.15`

3) Train
- Command: `python src/train.py --config configs/train.yaml`
- Optional quick run: `python src/train.py --config configs/train.yaml --epochs 10 --limit_train 100`

4) Inference (Python helper)
- Example:
```python
from src.infer import InferenceModel
m = InferenceModel('weights/best.pt', model_name='efficientnet_b0')
print(m.predict_path('path/to/image.jpg', topk=5))
```

5) Streamlit web app (with Gemini insights)
- Ensure `google-generativeai` is installed and API key available via env var `GEMINI_API_KEY` or `secrets/gemini_api_key.txt`
- Run: `streamlit run src/streamlit_app.py`

## Notes
- Check `configs/train.yaml` for hyperparameters.
- Best weights are saved to `weights/best.pt` automatically during training.
- Accuracy value is recorded in `weights/best.pt` under `best_metric`. If needed, you can extract it via a short Python command.
