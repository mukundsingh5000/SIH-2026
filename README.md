# TARANG — AI Sonar Intelligence for Marine Debris Detection
**Smart India Hackathon (SIH) — Problem Statement PS57**

TARANG is a real-time AI sonar intelligence platform that detects and classifies marine debris (aircraft, bottle, cylinder, human, net, pipe, wreck) from subsea acoustic sonar imagery using deep learning edge-scoring models.

---

## 🏗 Architecture & Data Flow

```
[Sonar Image File]
        │
        ▼
 React + Vite Frontend (Port 3000)
        │
        ▼  POST /api/predict
 FastAPI Backend Server (Port 8000)
        │
        ▼
 PyTorch GhostNet Inference Engine (ghostnet_scorer.pt)
        │
        ▼  Class Prediction + Confidence Score
 SQLite Database (tarang_predictions.db)
        │
        ▼  JSON Response
 React Real-Time Dashboard + Detection History
```

---

## 🏷 Class Mapping Reference

| Class ID | Class Name | Detection Status | Description |
| :--- | :--- | :--- | :--- |
| `0` | `background` | `detected: false` | No marine object detected (Clear seabed) |
| `1` | `aircraft` | `detected: true` | Submerged aircraft debris |
| `2` | `bottle` | `detected: true` | Plastic / glass bottle debris |
| `3` | `cylinder` | `detected: true` | Industrial cylinder / canister |
| `4` | `human` | `detected: true` | Human presence / diver detection |
| `5` | `net` | `detected: true` | Abandoned fishing net / ghost gear |
| `6` | `pipe` | `detected: true` | Underwater pipeline / metal pipe |
| `7` | `wreck` | `detected: true` | Shipwreck / structural wreckage |

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.9+ installed
- Node.js v18+ installed

---

### 2. Backend Setup (FastAPI)

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Start FastAPI server
uvicorn app:app --reload --port 8000
```
- API Health Check: `http://localhost:8000/api/health`
- Swagger API Docs: `http://localhost:8000/docs`

---

### 3. Frontend Setup (React + Vite)

Open a second terminal window:

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Vite dev server
npm run dev
```
- Access Frontend UI at: `http://localhost:3000` (or `http://localhost:5173`)

---

## 🛰 API Endpoints

- `GET /api/health` — Check server status & verify PyTorch model is loaded.
- `POST /api/predict` — Upload sonar image file (`file`) for inference.
- `GET /api/predictions` — Fetch historical detection records from SQLite database.

---

## 📁 File Structure

```
TARANG/
├── backend/
│   ├── model/
│   │   └── ghostnet_scorer.pt    # Trained PyTorch CNN model checkpoint
│   ├── app.py                    # FastAPI server & route handlers
│   ├── predictor.py              # Model loader & preprocessing inference
│   ├── database.py               # SQLite database interface
│   └── requirements.txt          # Minimal backend dependencies
│
├── frontend/
│   ├── src/
│   │   ├── services/
│   │   │   └── api.js            # API client for FastAPI endpoints
│   │   ├── App.jsx               # Main React dashboard component
│   │   ├── index.css             # Oceanic dark mode styling system
│   │   └── main.jsx              # Vite React entrypoint
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
│
├── train_cnn.py                  # Training reference code
├── demo_infer.py                 # Original inference reference code
├── .gitignore
└── README.md
```
