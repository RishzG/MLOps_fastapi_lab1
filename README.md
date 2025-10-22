# FastAPI Iris Classifier

Small MLOps lab that wraps a scikit-learn decision tree in a FastAPI service with a browser dashboard for exploring predictions on the Iris dataset.

## Project Structure

- `src/data.py` - loads the Iris dataset and creates deterministic train/test splits.
- `src/train.py` - fits a depth-limited decision tree and persists it to `model/iris_model.pkl`.
- `src/predict.py` - caches the trained model and exposes helpers for class predictions and class probabilities.
- `src/main.py` - FastAPI application that validates inputs, serves a UI, and offers `/predict` and `/predict-proba` endpoints.
- `src/templates/dashboard.html` - Jinja2 template rendered by FastAPI; submits measurements via HTML form and renders probability tables.
- `requirements.txt` - Python dependencies for the service.

## Features

- FastAPI JSON API with input validation (`/predict`, `/predict-proba`).
- Inline logging for successful predictions and validation failures.
- Responsive dashboard rendered by FastAPI with probability tables and a Start Over control for quick resets.

## Getting Started

1. **Create/activate a virtual environment** (example on Windows PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the model** (writes `model/iris_model.pkl`):
   ```bash
   python -m src.train
   ```
4. **Run the FastAPI service**:
   ```bash
   uvicorn src.main:app --reload
   ```
5. **Open the dashboard**: visit [http://localhost:8000/dashboard](http://localhost:8000/dashboard) and submit measurements to see predictions and probability tables.

## API Usage

Send the same four measurements as JSON to the REST API.

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

For probability details:

```bash
curl -X POST http://localhost:8000/predict-proba \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

The response contains the predicted class, its human-readable label, and the probability assigned to every Iris species.

```

## Next Steps

- Swap in a different model or dataset for experimentation.
- Extend the API with model-metrics endpoints or background retraining jobs.
- Containerize the service for deployment in a fuller MLOps workflow.
