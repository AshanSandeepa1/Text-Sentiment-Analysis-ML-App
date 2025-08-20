# main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, constr
import joblib
import json
import re
import logging
from typing import List

# -----------------------
# Logging setup to file + console
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("api.log", mode='a'),  # append logs to api.log
        logging.StreamHandler()                   # also print to console
    ]
)

# -----------------------
# Load model and label mapping
# -----------------------
model = joblib.load("model.pkl")
with open("id2label.json", "r") as f:
    id2label = json.load(f)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting tweet sentiment",
    version="1.2"
)

# -----------------------
# CORS middleware
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Templates
# -----------------------
templates = Jinja2Templates(directory="templates")

# -----------------------
# Helper: Text preprocessing
# -----------------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"@\w+", "", text)              # remove mentions
    text = re.sub(r"[^a-z0-9\s]", "", text)      # remove special chars & emojis
    text = re.sub(r"\s+", " ", text).strip()     # remove extra spaces
    return text

# -----------------------
# Pydantic schemas
# -----------------------
class PredictionInput(BaseModel):
    text: constr(strip_whitespace=True, min_length=1, max_length=5000)

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float

class BatchPredictionInput(BaseModel):
    texts: List[constr(strip_whitespace=True, min_length=1, max_length=5000)]

class BatchPredictionOutputItem(BaseModel):
    text: str
    prediction: str
    confidence: float

class BatchPredictionOutput(BaseModel):
    results: List[BatchPredictionOutputItem]

# -----------------------
# Endpoints
# -----------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Sentiment Analysis API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        cleaned_text = preprocess_text(input_data.text)
        if not cleaned_text or not any(c.isalpha() for c in cleaned_text):
            raise HTTPException(
                status_code=400,
                detail="Input must contain at least one alphabet character."
            )

        pred = int(model.predict([cleaned_text])[0])
        prob = model.predict_proba([cleaned_text])[0]
        confidence = round(float(max(prob)), 2)
        label_name = id2label[str(pred)]

        logging.info(f"Single predict | input: {input_data.text} | cleaned: {cleaned_text} | prediction: {label_name} | confidence: {confidence}")

        return PredictionOutput(prediction=label_name, confidence=confidence)

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Prediction failed | input: {input_data.text} | Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionOutput)
def predict_batch(input_data: BatchPredictionInput):
    results = []
    for text in input_data.texts:
        cleaned_text = preprocess_text(text)
        if not cleaned_text or not any(c.isalpha() for c in cleaned_text):
            results.append(BatchPredictionOutputItem(
                text=text,
                prediction="invalid input",
                confidence=0.0
            ))
            logging.warning(f"Batch prediction invalid input: {text}")
            continue

        try:
            pred = int(model.predict([cleaned_text])[0])
            prob = model.predict_proba([cleaned_text])[0]
            confidence = round(float(max(prob)), 2)
            label_name = id2label[str(pred)]

            results.append(BatchPredictionOutputItem(
                text=text,
                prediction=label_name,
                confidence=confidence
            ))

            logging.info(f"Batch predict | input: {text} | prediction: {label_name} | confidence: {confidence}")

        except Exception as e:
            results.append(BatchPredictionOutputItem(
                text=text,
                prediction="error",
                confidence=0.0
            ))
            logging.error(f"Batch prediction failed | input: {text} | Error: {str(e)}")

    return BatchPredictionOutput(results=results)


@app.get("/model-info")
def model_info():
    return {
        "model_type": "LogisticRegression + TF-IDF Pipeline",
        "problem_type": "binary classification",
        "features": ["text"]
    }
