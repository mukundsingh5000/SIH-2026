import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predictor import SonarPredictor
import database

predictor_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor_instance
    database.init_db()
    
    # Path to model file
    model_path = os.path.join(os.path.dirname(__file__), "model", "ghostnet_scorer.pt")
    if not os.path.exists(model_path):
        root_model = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ghostnet_scorer.pt")
        if os.path.exists(root_model):
            model_path = root_model

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print(f"[Lifespan] Loading model from: {model_path}")
    predictor_instance = SonarPredictor(model_path)
    yield


app = FastAPI(
    title="TARANG - AI Sonar Intelligence API",
    description="Marine Debris Detection API for SIH PS57",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration to allow React Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "TARANG Marine Debris Detection API",
        "model_loaded": predictor_instance is not None
    }


@app.post("/api/predict")
async def predict_sonar(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file received")
        
    try:
        prediction = predictor_instance.predict(contents)
        saved_record = database.save_prediction(
            class_id=prediction["class_id"],
            class_name=prediction["class_name"],
            confidence=prediction["confidence"],
            detected=prediction["detected"]
        )
        return {
            "success": True,
            "id": saved_record["id"],
            "timestamp": saved_record["timestamp"],
            "class_id": prediction["class_id"],
            "class_name": prediction["class_name"],
            "confidence": prediction["confidence"],
            "detected": prediction["detected"]
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/api/predictions")
def get_predictions():
    return database.get_all_predictions()


@app.get("/api/analytics")
def get_analytics():
    return database.get_analytics_summary()

