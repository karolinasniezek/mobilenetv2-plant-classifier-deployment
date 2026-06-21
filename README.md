# Plant Seedlings Classification System

## Description

This project is an end-to-end machine learning application for plant seedling classification.

The solution consists of:

- TensorFlow/Keras model trained using MobileNetV2 transfer learning
- FastAPI backend exposing a prediction endpoint
- Streamlit frontend for image upload and inference

The application allows users to upload an image of a plant seedling and receive:

- predicted class
- confidence score

---

## Technologies

```text
Python
TensorFlow
Keras
MobileNetV2
NumPy
FastAPI
Uvicorn
Streamlit
Requests
```

---

## Supported Classes

```text
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

---

## Model Architecture

### Base Model

```python
MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
```

### Classification Head

```python
GlobalAveragePooling2D()

Dense(128)

Dense(12, activation="softmax")
```

### Input Shape

```python
(224, 224, 3)
```

### Output

```python
12 classes
```

---

## Project Structure

```text
plant-seedlings-classifier/

├── data/
│   ├── plants_train/
│   └── plants_test/
│
├── src/
│   ├── train_model.py
│   ├── app.py
│   └── app_frontend.py
│
├── plant_seedlings_model.h5
├── requirements.txt
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/plant-seedlings-classifier.git](https://github.com/karolinasniezek/mobilenetv2-plant-classifier-deployment.git

cd Mobilenetv2PlantClassifierDeployment
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

#### Linux / macOS

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---


## Training

### Train Model

```bash
python src/train_model.py
```

### Output Model

```text
plant_seedlings_model.h5
```

---

## Running the Backend

### Start FastAPI

```bash
uvicorn src.app:app --reload
```

### API URL

```text
http://localhost:8000
```

### Swagger Documentation

```text
http://localhost:8000/docs
```

### Prediction Endpoint

```http
POST /predict
```

---

## Running the Frontend

### Start Streamlit

```bash
streamlit run src/app_frontend.py
```

### Frontend URL

```text
http://localhost:8501
```

---

## API Example

### Request

```bash
curl -X POST \
"http://localhost:8000/predict" \
-H "Content-Type: multipart/form-data" \
-F "file=@sample.jpg"
```

### Response

```json
{
  "class": "Maize",
  "confidence": 0.9734
}
```

---

## Inference Pipeline

```text
Image Upload
      |
      v
Streamlit Frontend
      |
      v
FastAPI Backend
      |
      v
TensorFlow Model
      |
      v
Prediction
      |
      v
Class + Confidence Score
```

---

## Reproducibility

### Random Seeds

```python
seed = 42

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
keras.utils.set_random_seed(seed)
```

### Deterministic Operations

```python
os.environ["TF_DETERMINISTIC_OPS"] = "1"

tf.config.experimental.enable_op_determinism()
```

---

## Features

- Image classification using MobileNetV2
- Transfer learning approach
- FastAPI REST API
- Streamlit user interface
- Confidence score reporting
- Reproducible training pipeline
- Interactive image upload
- Real-time prediction

---

## Future Improvements

```text
Docker support
CI/CD pipeline
MLflow integration
Cloud deployment
Model monitoring
Model versioning
Batch inference
```

---

Machine Learning / Deep Learning Portfolio Project

This project demonstrates the complete machine learning lifecycle, including model training, deployment through a REST API, and integration with a user-facing web application.
