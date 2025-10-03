from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import joblib
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings:
    MODEL_PATH = r"models\best_model.pth"
    SCALER_PATH = r"models\exoplanet_scaler.pkl"
    LABEL_ENCODER_PATH = r"models\exoplanet_label_encoder.pkl"
    MODEL_INFO_PATH = r"models\model_info.pkl"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

settings = Settings()

class ExoplanetInput(BaseModel):
    # Identifiers
    kepid: Optional[int] = Field(None, description="Kepler ID")
    kepoi_name: Optional[str] = Field(None, description="KOI Name")

    # Planet characteristics
    koi_score: Optional[float] = Field(None, description="Disposition score")
    koi_period: Optional[float] = Field(None, description="Orbital period (days)")
    koi_duration: Optional[float] = Field(None, description="Transit duration (hours)")
    koi_depth: Optional[float] = Field(None, description="Transit depth (ppm)")
    koi_time0bk: Optional[float] = Field(None, description="Transit epoch")
    koi_prad: Optional[float] = Field(None, description="Planetary radius (Earth radii)")
    koi_ror: Optional[float] = Field(None, description="Planet-star radius ratio")
    koi_dor: Optional[float] = Field(None, description="Planet-star distance over stellar radius")
    koi_impact: Optional[float] = Field(None, description="Impact parameter")
    koi_incl: Optional[float] = Field(None, description="Inclination (degrees)")

    # Stellar characteristics
    koi_srad: Optional[float] = Field(None, description="Stellar radius (solar radii)")
    koi_smass: Optional[float] = Field(None, description="Stellar mass (solar masses)")
    koi_steff: Optional[float] = Field(None, description="Stellar effective temperature (K)")
    koi_slogg: Optional[float] = Field(None, description="Stellar surface gravity")
    koi_smet: Optional[float] = Field(None, description="Stellar metallicity")
    koi_kepmag: Optional[float] = Field(None, description="Kepler magnitude")

    # False positive flags
    koi_fpflag_nt: Optional[int] = Field(None, description="Not transit-like flag")
    koi_fpflag_ss: Optional[int] = Field(None, description="Stellar eclipse flag")
    koi_fpflag_co: Optional[int] = Field(None, description="Centroid offset flag")
    koi_fpflag_ec: Optional[int] = Field(None, description="Ephemeris match flag")

    # Error measurements
    koi_period_err1: Optional[float] = Field(None, description="Period error (+)")
    koi_period_err2: Optional[float] = Field(None, description="Period error (-)")
    koi_duration_err1: Optional[float] = Field(None, description="Duration error (+)")
    koi_duration_err2: Optional[float] = Field(None, description="Duration error (-)")
    koi_depth_err1: Optional[float] = Field(None, description="Depth error (+)")
    koi_depth_err2: Optional[float] = Field(None, description="Depth error (-)")
    koi_prad_err1: Optional[float] = Field(None, description="Radius error (+)")
    koi_prad_err2: Optional[float] = Field(None, description="Radius error (-)")
    koi_steff_err1: Optional[float] = Field(None, description="Temperature error (+)")
    koi_steff_err2: Optional[float] = Field(None, description="Temperature error (-)")

    class Config:
        json_schema_extra = {
            "example": {
                "kepid": 10797460,
                "kepoi_name": "K00752.01",
                "koi_period": 9.48803557,
                "koi_duration": 2.9575,
                "koi_depth": 615.8,
                "koi_time0bk": 170.53875,
                "koi_prad": 2.26,
                "koi_ror": 0.022344,
                "koi_dor": 24.81,
                "koi_impact": 0.146,
                "koi_incl": 89.66,
                "koi_srad": 0.927,
                "koi_smass": 0.919,
                "koi_steff": 5455,
                "koi_slogg": 4.467,
                "koi_smet": 0.14,
                "koi_kepmag": 15.347,
                "koi_fpflag_nt": 0,
                "koi_fpflag_ss": 0,
                "koi_fpflag_co": 0,
                "koi_fpflag_ec": 0
            }
        }


class PredictionOutput(BaseModel):
    kepid: Optional[int] = None
    kepoi_name: Optional[str] = None
    predicted_disposition: str = Field(..., description="Predicted class: CONFIRMED, CANDIDATE, or FALSE POSITIVE")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for prediction")
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    prediction_timestamp: str


class BatchPredictionRequest(BaseModel):
    data: List[ExoplanetInput] = Field(..., description="List of exoplanet observations")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {
                        "kepid": 10797460,
                        "kepoi_name": "K00752.01",
                        "koi_period": 9.48803557,
                        "koi_duration": 2.9575,
                        "koi_depth": 615.8,
                        "koi_prad": 2.26,
                        "koi_steff": 5455
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionOutput]
    batch_size: int
    processing_time_ms: float
    model_info: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str


class ImprovedClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int, dropout_rate: float = 0.3):
        super(ImprovedClassifier, self).__init__()

        self.layer_1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer_2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer_3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_rate * 0.67)

        self.layer_4 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.layer_2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.layer_3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        x = self.layer_4(x)
        return x

class ModelLoader:
    _instance = None
    _model = None
    _scaler = None
    _label_encoder = None
    _model_info = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def load(self):
        try:
            logger.info("Loading model artifacts...")

            self._scaler = joblib.load(settings.SCALER_PATH)
            self._label_encoder = joblib.load(settings.LABEL_ENCODER_PATH)

            with open(settings.MODEL_INFO_PATH, 'rb') as f:
                self._model_info = pickle.load(f)

            self._device = torch.device(settings.DEVICE)

            input_size = self._model_info['input_size']
            num_classes = self._model_info['num_classes']

            self._model = ImprovedClassifier(input_size, num_classes)

            checkpoint = torch.load(
                settings.MODEL_PATH,
                map_location=self._device,
                weights_only=False
            )
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.to(self._device)
            self._model.eval()

            logger.info(f"Model loaded successfully on {self._device}")
            logger.info(f"Input size: {input_size}, Classes: {num_classes}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def scaler(self):
        if self._scaler is None:
            self.load()
        return self._scaler

    @property
    def label_encoder(self):
        if self._label_encoder is None:
            self.load()
        return self._label_encoder

    @property
    def model_info(self):
        if self._model_info is None:
            self.load()
        return self._model_info

    @property
    def device(self):
        if self._device is None:
            self.load()
        return self._device


app = FastAPI(
    title="Exoplanet Classification API",
    description="Predict exoplanet dispositions (CONFIRMED/CANDIDATE/FALSE POSITIVE) from observational features",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_loader = ModelLoader()


def preprocess_input(input_data: List[ExoplanetInput]) -> pd.DataFrame:
    try:
        df = pd.DataFrame([item.model_dump() for item in input_data])

        logger.info(f"Received DataFrame shape: {df.shape}")

        columns_to_drop = ['kepoi_name', 'kepid']
        df = df.drop(columns=columns_to_drop, errors='ignore')

        expected_features = model_loader.model_info['feature_names']
        logger.info(f"Expected {len(expected_features)} features")

        for feature in expected_features:
            if feature not in df.columns:
                logger.warning(f"Missing feature: {feature}, adding with value 0")
                df[feature] = 0.0

        df = df[expected_features]

        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            logger.info(f"Filling missing values in: {missing_cols}")
            for col in missing_cols:
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df[col] = df[col].fillna(median_val)

        df = df.astype(float)

        if df.isnull().any().any():
            logger.warning("NaN values detected, replacing with 0")
            df = df.fillna(0.0)

        if np.isinf(df.values).any():
            logger.warning("Infinite values detected, replacing with 0")
            df = df.replace([np.inf, -np.inf], 0.0)

        logger.info(f"Preprocessed shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise


def make_predictions(features_df: pd.DataFrame) -> tuple:
    try:
        features_scaled = model_loader.scaler.transform(features_df.values)

        if np.isnan(features_scaled).any():
            logger.warning("NaN detected after scaling, replacing with 0")
            features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(model_loader.device)

        with torch.no_grad():
            outputs = model_loader.model(features_tensor)

            if torch.isnan(outputs).any():
                logger.error("NaN in model outputs")
                raise ValueError("Model produced NaN outputs")

            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

        predictions_np = predictions.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()

        return predictions_np, probabilities_np

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    try:
        model_loader.load()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Exoplanet Classification API",
        "description": "Predict exoplanet dispositions from observational features",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict_single": "/predict/single",
            "predict_batch": "/predict"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader._model is not None,
        device=str(settings.DEVICE),
        timestamp=datetime.now().isoformat()
    )


@app.get("/model-info", tags=["Model"])
async def get_model_info():
    try:
        info = model_loader.model_info
        return {
            "input_size": info['input_size'],
            "num_classes": info['num_classes'],
            "classes": list(model_loader.label_encoder.classes_),
            "feature_names": info['feature_names'],
            "best_val_accuracy": info.get('best_val_acc'),
            "test_accuracy": info.get('test_acc'),
            "training_time": info.get('training_time'),
            "device": str(model_loader.device)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict(request: BatchPredictionRequest):
    start_time = datetime.now()

    try:
        if not request.data or len(request.data) == 0:
            raise HTTPException(status_code=400, detail="No data provided")

        logger.info(f"📊 Received {len(request.data)} samples for prediction")

        kepids = [item.kepid for item in request.data]
        kepoi_names = [item.kepoi_name for item in request.data]

        features_df = preprocess_input(request.data)
        predictions, probabilities = make_predictions(features_df)

        results = []
        for i, (pred_idx, probs) in enumerate(zip(predictions, probabilities)):
            predicted_label = model_loader.label_encoder.inverse_transform([pred_idx])[0]

            confidence = float(probs[pred_idx])
            if np.isnan(confidence) or np.isinf(confidence):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            prob_dict = {}
            for j in range(len(model_loader.label_encoder.classes_)):
                class_name = model_loader.label_encoder.inverse_transform([j])[0]
                prob_val = float(probs[j])
                if np.isnan(prob_val) or np.isinf(prob_val):
                    prob_val = 0.0
                prob_val = max(0.0, min(1.0, prob_val))
                prob_dict[class_name] = prob_val

            result = PredictionOutput(
                kepid=kepids[i],
                kepoi_name=kepoi_names[i],
                predicted_disposition=predicted_label,
                confidence=confidence,
                probabilities=prob_dict,
                prediction_timestamp=datetime.now().isoformat()
            )
            results.append(result)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(f"Predictions complete in {processing_time:.2f}ms")

        return BatchPredictionResponse(
            predictions=results,
            batch_size=len(request.data),
            processing_time_ms=processing_time,
            model_info={
                "device": str(model_loader.device),
                "model_version": "2.0.0"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/single", response_model=PredictionOutput, tags=["Prediction"])
async def predict_single(observation: ExoplanetInput):
    batch_request = BatchPredictionRequest(data=[observation])
    batch_response = await predict(batch_request)
    return batch_response.predictions[0]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
