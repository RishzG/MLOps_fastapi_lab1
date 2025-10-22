import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

from fastapi import FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError

from predict import predict_data, predict_probabilities

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

IRIS_CLASS_NAMES = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica",
}

DEFAULT_FORM_VALUES = {
    "sepal_length": "",
    "sepal_width": "",
    "petal_length": "",
    "petal_width": "",
}


class IrisData(BaseModel):
    """Validated Iris feature payload."""

    sepal_length: float = Field(..., gt=0, lt=10, description="Sepal length in centimeters")
    sepal_width: float = Field(..., gt=0, lt=10, description="Sepal width in centimeters")
    petal_length: float = Field(..., gt=0, lt=10, description="Petal length in centimeters")
    petal_width: float = Field(..., gt=0, lt=10, description="Petal width in centimeters")


class IrisResponse(BaseModel):
    response: int


class IrisProbabilityResponse(BaseModel):
    predicted_class: int
    label: str
    probabilities: Dict[str, float]


def _format_validation_error(validation_error: ValidationError) -> str:
    issues = []
    for error in validation_error.errors():
        field = error.get("loc", ["value"])[-1]
        message = error.get("msg", "Invalid value")
        issues.append(f"{field}: {message}")
    return "; ".join(issues)


def _format_probabilities(probabilities: Iterable[float]) -> Dict[str, float]:
    return {
        IRIS_CLASS_NAMES.get(idx, str(idx)): float(prob)
        for idx, prob in enumerate(probabilities)
    }


def _probability_rows(probabilities: Iterable[float]):
    return [
        {"label": IRIS_CLASS_NAMES.get(idx, str(idx)), "value": float(prob)}
        for idx, prob in enumerate(probabilities)
    ]


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}


@app.get("/dashboard", response_class=HTMLResponse, status_code=status.HTTP_200_OK)
async def render_dashboard(
    request: Request,
    result: Optional[int] = None,
    label: Optional[str] = None,
    error: Optional[str] = None,
):
    context = {
        "request": request,
        "result": result,
        "label": label,
        "error": error,
        "probabilities": None,
        "form_values": DEFAULT_FORM_VALUES.copy(),
    }
    return templates.TemplateResponse("dashboard.html", context)


@app.post("/dashboard", response_class=HTMLResponse, status_code=status.HTTP_200_OK)
async def handle_dashboard_submit(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...),
):
    raw_form_values = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    try:
        iris_payload = IrisData(**raw_form_values)
        features = [[
            iris_payload.sepal_length,
            iris_payload.sepal_width,
            iris_payload.petal_length,
            iris_payload.petal_width,
        ]]
        prediction = predict_data(features)
        probabilities = predict_probabilities(features)
        predicted_class = int(prediction[0])
        probability_vector = probabilities[0]
        logger.info("Dashboard prediction succeeded for %s -> class %s", iris_payload.model_dump(), predicted_class)
        context = {
            "request": request,
            "result": predicted_class,
            "label": IRIS_CLASS_NAMES.get(predicted_class, str(predicted_class)),
            "error": None,
            "probabilities": _probability_rows(probability_vector),
            "form_values": {key: str(value) for key, value in iris_payload.model_dump().items()},
        }
    except ValidationError as validation_error:
        message = _format_validation_error(validation_error)
        logger.warning("Dashboard validation failed: %s", message)
        context = {
            "request": request,
            "result": None,
            "label": None,
            "error": message,
            "probabilities": None,
            "form_values": {key: str(value) for key, value in raw_form_values.items()},
        }
    except Exception as exc:
        logger.exception("Dashboard prediction error")
        context = {
            "request": request,
            "result": None,
            "label": None,
            "error": str(exc),
            "probabilities": None,
            "form_values": {key: str(value) for key, value in raw_form_values.items()},
        }
    return templates.TemplateResponse("dashboard.html", context)


@app.post("/predict", response_model=IrisResponse)
async def predict_iris(iris_features: IrisData):
    try:
        payload = iris_features.model_dump()
        features = [[
            iris_features.sepal_length,
            iris_features.sepal_width,
            iris_features.petal_length,
            iris_features.petal_width,
        ]]
        prediction = predict_data(features)
        predicted_class = int(prediction[0])
        logger.info("API prediction succeeded for %s -> class %s", payload, predicted_class)
        return IrisResponse(response=predicted_class)

    except Exception as error:
        logger.exception("Prediction failed for %s", iris_features)
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/predict-proba", response_model=IrisProbabilityResponse)
async def predict_iris_with_probabilities(iris_features: IrisData):
    try:
        payload = iris_features.model_dump()
        features = [[
            iris_features.sepal_length,
            iris_features.sepal_width,
            iris_features.petal_length,
            iris_features.petal_width,
        ]]
        prediction = predict_data(features)
        probabilities = predict_probabilities(features)
        predicted_class = int(prediction[0])
        probability_vector = probabilities[0]
        logger.info(
            "Probability request succeeded for %s -> class %s", payload, predicted_class
        )
        return IrisProbabilityResponse(
            predicted_class=predicted_class,
            label=IRIS_CLASS_NAMES.get(predicted_class, str(predicted_class)),
            probabilities=_format_probabilities(probability_vector),
        )
    except Exception as error:
        logger.exception("Probability prediction failed for %s", iris_features)
        raise HTTPException(status_code=500, detail=str(error))
