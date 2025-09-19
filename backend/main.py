from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
import logging
from datetime import datetime, timedelta
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AgriTech API",
    description="Comprehensive API for crop recommendation, weather forecasting, and soil type prediction",
    version="1.0.0"
)

# Define the 10 core features for crop prediction
CORE_FEATURES = [
    'N', 'P', 'K', 'temperature', 'humidity', 
    'ph', 'rainfall', 'log_rainfall', 'npk_sum', 'ph_bin_enc'
]

# Crop water requirement classification
CROP_WATER_REQUIREMENTS = {
    'Low Water': [
        'mothbeans', 'horsegram', 'chickpea', 'lentil', 'blackgram', 'mungbean',
        'greengram', 'cowpea', 'pigeonpeas', 'ragi', 'jowar', 'sesamum',
        'sunflower', 'castor', 'mustard', 'linseed', 'safflower'
    ],
    'Medium Water': [
        'wheat', 'barley', 'oats', 'maize', 'sorghum', 'pearlmillet',
        'finger millet', 'soyabean', 'groundnut', 'cotton', 'tobacco',
        'chilli', 'turmeric', 'ginger', 'garlic', 'onion', 'potato'
    ],
    'High Water': [
        'rice', 'sugarcane', 'banana', 'coconut', 'arecanut', 'betelvine',
        'jute', 'kenaf', 'sunhemp', 'tea', 'coffee', 'rubber',
        'cardamom', 'blackpepper', 'vanilla', 'cocoa', 'fruit trees'
    ],
    'Very High Water': [
        'watermelon', 'muskmelon', 'cucumber', 'pumpkin', 'bottlegourd',
        'ridgegourd', 'spongegourd', 'pointedgourd', 'ivygourd',
        'tomato', 'brinjal', 'okra', 'cauliflower', 'cabbage',
        'spinach', 'lettuce', 'celery', 'radish', 'carrot', 'beetroot'
    ]
}

# Soil CNN Model Definition
class SoilCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SoilCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 16 * 16)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load model artifacts
MODEL_DIR = "model_artifacts"
SOIL_MODEL_PATH = os.path.join(MODEL_DIR, "best_soil_cnn.pth")
SOIL_TYPES = ['Black Soil', 'Cinder Soil', 'Laterite Soil', 'Peat Soil', 'Yellow Soil']

try:
    # Load crop prediction model
    crop_model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    feature_medians = joblib.load(os.path.join(MODEL_DIR, "feature_medians.joblib"))
    logger.info("Crop model artifacts loaded successfully")
except Exception as e:
    logger.error(f"Failed to load crop model artifacts: {str(e)}")
    crop_model = None
    scaler = None
    label_encoder = None
    feature_medians = {}

try:
    # Load soil CNN model
    soil_model = SoilCNN(num_classes=5)
    soil_model.load_state_dict(torch.load(SOIL_MODEL_PATH, map_location=torch.device('cpu')))
    soil_model.eval()
    logger.info("Soil CNN model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load soil CNN model: {str(e)}")
    soil_model = None

# Image transformation for soil prediction
soil_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Pydantic models
class CropPredictionRequest(BaseModel):
    N: Optional[float] = None
    P: Optional[float] = None
    K: Optional[float] = None
    ph: Optional[float] = None
    latitude: float
    longitude: float
    soil_type: Optional[str] = "Black Soil"

class CropPrediction(BaseModel):
    crop: str
    confidence: float
    water_requirement: str

class WaterGroup(BaseModel):
    water_requirement: str
    crops: List[CropPrediction]

class PredictionResponse(BaseModel):
    predicted_crop: str
    confidence: float
    water_scarcity_level: str
    top_5_predictions: List[CropPrediction]
    crops_by_water_requirement: List[WaterGroup]
    recommendations: List[str]
    weather_forecast: Dict
    soil_type: str

class HealthResponse(BaseModel):
    status: str
    crop_model_loaded: bool
    soil_model_loaded: bool
    num_classes: Optional[int] = None
    features: List[str]
    weather_module: bool

class WeatherRequest(BaseModel):
    latitude: float
    longitude: float
    months: Optional[int] = 12

class WaterScarcityResponse(BaseModel):
    water_scarcity_level: str
    months_affected: List[str]
    recommendations: List[str]
    forecast_summary: Dict

class SoilPredictionResponse(BaseModel):
    soil_type: str
    confidence: float

# Helper Functions
def predict_soil_type_from_image(image_file: UploadFile) -> tuple[str, float]:
    """Predict soil type from uploaded image"""
    if soil_model is None:
        raise HTTPException(status_code=503, detail="Soil prediction model not loaded")
    
    try:
        # Read and process image
        image_bytes = image_file.file.read()
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image = soil_transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = soil_model(image)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return SOIL_TYPES[predicted.item()], float(confidence)
    except Exception as e:
        logger.error(f"Soil prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Soil prediction failed: {str(e)}")

def get_water_requirement(crop_name: str) -> str:
    """Get water requirement category for a crop"""
    for requirement, crops in CROP_WATER_REQUIREMENTS.items():
        if crop_name.lower() in [c.lower() for c in crops]:
            return requirement
    return "Medium Water"

def categorize_crops_by_water(crop_predictions: List[dict]) -> List[WaterGroup]:
    """Categorize crops by their water requirements"""
    water_groups = {}
    
    for crop in crop_predictions:
        water_req = get_water_requirement(crop['crop'])
        if water_req not in water_groups:
            water_groups[water_req] = []
        water_groups[water_req].append(crop)
    
    for water_req in water_groups:
        water_groups[water_req].sort(key=lambda x: x['confidence'], reverse=True)
    
    return [
        WaterGroup(water_requirement=req, crops=crops)
        for req, crops in water_groups.items()
    ]

def generate_enhanced_synthetic_weather(latitude: float, longitude: float, months: int = 60) -> pd.DataFrame:
    """Generate synthetic weather data based on latitude and longitude"""
    rng = pd.date_range(datetime.now().replace(day=1), periods=months, freq="ME")
    t = np.arange(months)
    
    base_temp = 30 - (abs(latitude) * 0.5)
    base_rainfall = 100 + (abs(latitude) * 2)
    
    rainfall = (
        base_rainfall
        + 80 * np.sin(2 * np.pi * t / 12)
        + np.random.normal(0, 20, months)
    )
    
    temperature = (
        base_temp
        + 8 * np.sin(2 * np.pi * (t + 3) / 12)
        + np.random.normal(0, 2, months)
    )
    
    humidity = (
        60
        + 15 * np.sin(2 * np.pi * (t + 1) / 12)
        + 0.1 * rainfall
        + np.random.normal(0, 5, months)
    )
    
    return pd.DataFrame({
        "ds": rng, 
        "y": rainfall, 
        "temp": temperature, 
        "humidity": humidity
    })

def train_and_forecast_weather(df, periods=12):
    """Train Prophet model and forecast"""
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df[["ds", "y"]])
    future = model.make_future_dataframe(periods=periods, freq="ME")
    return model.predict(future)

def classify_water_scarcity(row):
    """Classify water scarcity level"""
    r = row["Rainfall (mm)"]
    h = row["Humidity (%)"]
    if r < 80 and h < 55:
        return "High"
    elif r < 120 or h < 60:
        return "Medium"
    else:
        return "Low"

def predict_water_scarcity(latitude: float, longitude: float, months: int = 12):
    """Predict water scarcity for given location"""
    historical_data = generate_enhanced_synthetic_weather(latitude, longitude, 60)
    forecast = train_and_forecast_weather(historical_data, months)
    
    extended_data = generate_enhanced_synthetic_weather(latitude, longitude, len(historical_data) + months)
    forecast = forecast.merge(extended_data[["ds", "temp", "humidity"]], on="ds", how="left")
    
    predictions = pd.DataFrame({
        "Date": forecast["ds"],
        "Rainfall (mm)": forecast["yhat"].round(1),
        "Temperature (°C)": forecast["temp"].round(1),
        "Humidity (%)": forecast["humidity"].round(1),
    })
    
    predictions["Water_Scarcity"] = predictions.apply(classify_water_scarcity, axis=1)
    return predictions.tail(months)

def calculate_derived_features(data: dict) -> dict:
    """Calculate derived features"""
    result = data.copy()
    
    if result.get('rainfall') is not None:
        result['log_rainfall'] = np.log1p(result['rainfall'])
    else:
        result['log_rainfall'] = np.log1p(feature_medians.get('rainfall', 100))
    
    n = result.get('N', feature_medians.get('N', 50))
    p = result.get('P', feature_medians.get('P', 50))
    k = result.get('K', feature_medians.get('K', 50))
    result['npk_sum'] = n + p + k
    
    ph_value = result.get('ph', feature_medians.get('ph', 6.5))
    if ph_value <= 5.5:
        ph_bin = 'acidic'
    elif ph_value <= 6.5:
        ph_bin = 'slightly_acidic'
    elif ph_value <= 7.5:
        ph_bin = 'neutral'
    else:
        ph_bin = 'alkaline'
    
    ph_bin_mapping = {'acidic': 0, 'slightly_acidic': 1, 'neutral': 2, 'alkaline': 3}
    result['ph_bin_enc'] = ph_bin_mapping.get(ph_bin, 2)
    
    return result

def prepare_features(data: dict) -> pd.DataFrame:
    """Prepare features for prediction"""
    full_data = calculate_derived_features(data)
    
    features = {}
    for feature in CORE_FEATURES:
        if feature in full_data:
            features[feature] = full_data[feature]
        else:
            features[feature] = feature_medians.get(feature, 0)
    
    return pd.DataFrame([features])[CORE_FEATURES]

def get_weather_data_for_location(latitude: float, longitude: float) -> dict:
    """Get current weather data for location"""
    synthetic = generate_enhanced_synthetic_weather(latitude, longitude, 1)
    return {
        'temperature': float(synthetic['temp'].iloc[0]),
        'humidity': float(synthetic['humidity'].iloc[0]),
        'rainfall': float(synthetic['y'].iloc[0])
    }

# API Endpoints
@app.get("/")
async def root():
    return {"message": "AgriTech API - Crop Recommendation & Soil Type Prediction", "status": "healthy"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        crop_model_loaded=crop_model is not None,
        soil_model_loaded=soil_model is not None,
        num_classes=len(label_encoder.classes_) if label_encoder else None,
        features=CORE_FEATURES,
        weather_module=True
    )

@app.post("/predict-soil", response_model=SoilPredictionResponse)
async def predict_soil(image: UploadFile = File(...)):
    """Predict soil type from an uploaded image"""
    try:
        soil_type, confidence = predict_soil_type_from_image(image)
        return SoilPredictionResponse(soil_type=soil_type, confidence=confidence)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in predict-soil: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/predict-crop", response_model=PredictionResponse)
async def predict_crop_with_weather(
    image: UploadFile = File(None),
    N: Optional[float] = Query(None),
    P: Optional[float] = Query(None),
    K: Optional[float] = Query(None),
    ph: Optional[float] = Query(None),
    latitude: float = Query(...),
    longitude: float = Query(...),
    soil_type: Optional[str] = Query("Black Soil")
):
    """Predict optimal crop with optional image-based soil type prediction"""
    if crop_model is None:
        raise HTTPException(status_code=503, detail="Crop prediction model not loaded")
    
    try:
        # Predict soil type from image if provided
        if image:
            predicted_soil, soil_confidence = predict_soil_type_from_image(image)
            if predicted_soil not in SOIL_TYPES:
                raise HTTPException(status_code=400, detail=f"Predicted soil type '{predicted_soil}' not recognized")
            soil_type = predicted_soil
            logger.info(f"Predicted soil type: {soil_type} (Confidence: {soil_confidence:.2%})")
        
        # Get weather data
        weather_data = get_weather_data_for_location(latitude, longitude)
        
        # Prepare input data
        input_data = {
            'N': N, 'P': P, 'K': K, 'ph': ph,
            'temperature': weather_data['temperature'],
            'humidity': weather_data['humidity'],
            'rainfall': weather_data['rainfall'],
            'soil_type': soil_type
        }
        
        # Make crop prediction
        features_df = prepare_features(input_data)
        scaled_features = scaler.transform(features_df)
        prediction = crop_model.predict(scaled_features)[0]
        probabilities = crop_model.predict_proba(scaled_features)[0]
        
        # Decode prediction
        predicted_crop = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])
        
        # Get all predictions with water requirements
        all_predictions = []
        for i, crop_name in enumerate(label_encoder.classes_):
            crop_pred = {
                'crop': crop_name,
                'confidence': float(probabilities[i]),
                'water_requirement': get_water_requirement(crop_name)
            }
            all_predictions.append(crop_pred)
        
        # Get top 5 predictions
        top_5 = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)[:5]
        
        # Predict water scarcity
        water_scarcity = predict_water_scarcity(latitude, longitude, 6)
        scarcity_months = water_scarcity[water_scarcity['Water_Scarcity'] == 'High']['Date'].dt.strftime('%B %Y').tolist()
        scarcity_level = "High" if scarcity_months else "Low"
        
        # Categorize crops by water requirement
        water_groups = categorize_crops_by_water(all_predictions)
        
        # Generate recommendations
        recommendations = [
            f"Recommended crop: {predicted_crop} (Confidence: {confidence:.1%})",
            f"Location: Latitude {latitude}, Longitude {longitude}",
            f"Soil type: {soil_type}",
            f"Current weather: {weather_data['temperature']}°C, {weather_data['rainfall']}mm rainfall"
        ]
        
        if scarcity_level == "High":
            recommendations.extend([
                "HIGH WATER SCARCITY ALERT",
                f"Water scarcity expected in: {', '.join(scarcity_months)}",
                "Consider drought-resistant crops from Low Water category",
                "Implement water conservation practices",
                "Explore rainwater harvesting options"
            ])
        else:
            recommendations.append("Favorable water conditions expected")
        
        return PredictionResponse(
            predicted_crop=predicted_crop,
            confidence=confidence,
            water_scarcity_level=scarcity_level,
            top_5_predictions=top_5,
            crops_by_water_requirement=water_groups,
            recommendations=recommendations,
            weather_forecast=weather_data,
            soil_type=soil_type
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/water-scarcity", response_model=WaterScarcityResponse)
async def get_water_scarcity_analysis(request: WeatherRequest):
    """Get detailed water scarcity analysis for a location"""
    try:
        predictions = predict_water_scarcity(request.latitude, request.longitude, request.months)
        
        high_scarcity = predictions[predictions['Water_Scarcity'] == 'High']
        medium_scarcity = predictions[predictions['Water_Scarcity'] == 'Medium']
        
        scarcity_months = high_scarcity['Date'].dt.strftime('%B %Y').tolist()
        
        recommendations = []
        if not high_scarcity.empty:
            recommendations.extend([
                "HIGH WATER SCARCITY ALERT",
                "Recommended LOW WATER crops:",
                "  - mothbeans, horsegram, chickpea, lentil",
                "  - blackgram, mungbean, ragi, jowar",
                "  - sesamum, sunflower, mustard",
                "Water conservation practices:",
                "  - Drip irrigation system",
                "  - Mulching to reduce evaporation",
                "  - Rainwater harvesting",
                "  - Soil moisture monitoring"
            ])
        elif not medium_scarcity.empty:
            recommendations.extend([
                "MODERATE WATER SCARCITY",
                "Recommended MEDIUM WATER crops:",
                "  - wheat, barley, maize, soyabean",
                "  - cotton, groundnut, potato, onion",
                "Water management:",
                "  - Efficient irrigation scheduling",
                "  - Use of drought-tolerant varieties",
                "  - Soil conservation practices"
            ])
        else:
            recommendations.append("Favorable water conditions for all crop types")
        
        forecast_summary = {
            "avg_rainfall": predictions['Rainfall (mm)'].mean().round(1),
            "avg_temperature": predictions['Temperature (°C)'].mean().round(1),
            "total_high_scarcity_months": len(high_scarcity),
            "total_medium_scarcity_months": len(medium_scarcity)
        }
        
        return WaterScarcityResponse(
            water_scarcity_level="High" if not high_scarcity.empty else "Medium" if not medium_scarcity.empty else "Low",
            months_affected=scarcity_months,
            recommendations=recommendations,
            forecast_summary=forecast_summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Water scarcity analysis failed: {str(e)}")

@app.get("/crops")
async def get_available_crops():
    """Get list of all available crops with water requirements"""
    if label_encoder is None:
        raise HTTPException(status_code=503, detail="Crop model not loaded")
    
    crops_with_water = []
    for crop_name in label_encoder.classes_:
        crops_with_water.append({
            'crop': crop_name,
            'water_requirement': get_water_requirement(crop_name)
        })
    
    return {"crops": crops_with_water}

@app.get("/example-prediction")
async def get_example_prediction():
    """Get an example prediction with sample data"""
    if crop_model is None:
        raise HTTPException(status_code=503, detail="Crop model not loaded")
    
    example_request = CropPredictionRequest(
        N=90, P=42, K=43, ph=6.5,
        latitude=23.0775, longitude=76.8513,
        soil_type="Black Soil"
    )
    
    return await predict_crop_with_weather(
        image=None, N=example_request.N, P=example_request.P, K=example_request.K,
        ph=example_request.ph, latitude=example_request.latitude, longitude=example_request.longitude,
        soil_type=example_request.soil_type
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)