"""API REST ligera para inferencia de LemonGuard AI."""

from __future__ import annotations

import io
import logging
import math
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol, Sequence

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from src.manifest import load_manifest


LOGGER = logging.getLogger("lemonguard.api")
IMAGE_SIZE = (224, 224)
DEFAULT_CLASS_NAMES = ("diseased", "healthy")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "EfficientNetB0.keras"
_RELEASE_VERSION = re.compile(r"model-v((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))\Z")


class Predictor(Protocol):
    """Interfaz pequeña para sustituir el modelo por un doble en las pruebas."""

    def predict(self, image_bytes: bytes) -> Sequence[float]:
        """Devuelve las probabilidades en el orden de las clases del manifiesto."""


class KerasPredictor:
    """Carga el modelo una vez y reutiliza su preprocesamiento de entrada."""

    def __init__(self, model_path: Path) -> None:
        # KerasHub registra las capas serializadas del artefacto antes de cargarlo.
        import keras_hub  # noqa: F401
        import keras

        self._keras = keras
        self._model = keras.models.load_model(str(model_path), compile=False)

    def predict(self, image_bytes: bytes) -> Sequence[float]:
        image = self._keras.utils.load_img(
            io.BytesIO(image_bytes),
            target_size=IMAGE_SIZE,
        )
        array = self._keras.utils.img_to_array(image)

        import numpy as np

        batch = np.expand_dims(array, axis=0)
        return self._model.predict(batch, verbose=0)[0]


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0, le=100, description="Confianza porcentual, de 0 a 100.")
    model_version: str


def _configured_model_path(model_path: str | Path | None) -> Path:
    configured = model_path if model_path is not None else os.environ.get("MODEL_PATH")
    if configured is None:
        return DEFAULT_MODEL_PATH
    return Path(configured).expanduser()


def _resolve_model_metadata(
    model_path: Path,
    version_override: str | None,
    class_names_override: Sequence[str] | None,
) -> tuple[str, tuple[str, ...]]:
    configured_version = version_override or os.environ.get("MODEL_VERSION")
    version_from_manifest: str | None = None
    class_names = tuple(class_names_override) if class_names_override is not None else None

    manifest_path = model_path.parent / "model-manifest.json"
    if manifest_path.is_file():
        manifest = load_manifest(manifest_path)
        model_metadata = manifest["model"]
        version_from_manifest = model_metadata.get("version")
        if not version_from_manifest:
            release_tag = manifest["provenance"].get("release_tag")
            match = _RELEASE_VERSION.fullmatch(release_tag) if isinstance(release_tag, str) else None
            if match:
                version_from_manifest = match.group(1)

        if class_names is None:
            class_names = tuple(
                item["name"] for item in model_metadata["output"]["classes"]
            )

    if configured_version and version_from_manifest and configured_version != version_from_manifest:
        raise RuntimeError(
            "MODEL_VERSION no coincide con la versión registrada en model-manifest.json."
        )
    version = configured_version or version_from_manifest
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError(
            "No se pudo determinar la versión del modelo. Configure MODEL_VERSION o use "
            "un model-manifest.json con model.version o provenance.release_tag."
        )

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES
    if len(class_names) != 2 or any(not isinstance(name, str) or not name for name in class_names):
        raise RuntimeError("Se requieren exactamente las dos clases del clasificador LemonGuard AI.")
    if len(set(class_names)) != len(class_names):
        raise RuntimeError("Los nombres de clase del modelo deben ser únicos.")
    return version, tuple(class_names)


def _validate_image(image_bytes: bytes) -> None:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="El archivo de imagen está vacío.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(
            status_code=400,
            detail="El archivo recibido no contiene una imagen válida.",
        ) from exc


def _validated_probabilities(values: Sequence[float], class_count: int) -> list[float]:
    try:
        probabilities = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("El modelo devolvió probabilidades inválidas.") from exc

    if (
        len(probabilities) != class_count
        or any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities)
        or not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-3)
    ):
        raise ValueError("El modelo devolvió una salida softmax inválida.")
    return probabilities


def create_app(
    *,
    predictor: Predictor | None = None,
    model_path: str | Path | None = None,
    model_version: str | None = None,
    class_names: Sequence[str] | None = None,
) -> FastAPI:
    """Crea la aplicación; permite inyectar un predictor simulado en pruebas."""
    configured_path = _configured_model_path(model_path)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        resolved_version, resolved_classes = _resolve_model_metadata(
            configured_path,
            model_version,
            class_names,
        )
        application.state.predictor = (
            predictor if predictor is not None else KerasPredictor(configured_path)
        )
        application.state.model_version = resolved_version
        application.state.class_names = resolved_classes
        try:
            yield
        finally:
            application.state.predictor = None

    application = FastAPI(
        title="LemonGuard AI",
        description="Clasificación de imágenes de hojas de limón.",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health(request: Request) -> dict[str, Any]:
        loaded = getattr(request.app.state, "predictor", None) is not None
        if not loaded:
            raise HTTPException(status_code=503, detail="El modelo no está cargado.")
        return {
            "status": "ok",
            "model_loaded": True,
            "model_version": request.app.state.model_version,
        }

    @application.post("/predict", response_model=PredictionResponse)
    async def predict(request: Request, image: UploadFile = File(...)) -> PredictionResponse:
        image_bytes = await image.read()
        await image.close()
        _validate_image(image_bytes)

        runtime_predictor: Predictor = request.app.state.predictor
        classes: tuple[str, ...] = request.app.state.class_names
        try:
            raw_probabilities = await run_in_threadpool(
                runtime_predictor.predict,
                image_bytes,
            )
            probabilities = _validated_probabilities(raw_probabilities, len(classes))
        except Exception as exc:
            LOGGER.exception("Falló la inferencia de LemonGuard AI.")
            raise HTTPException(
                status_code=500,
                detail="No fue posible completar la inferencia.",
            ) from exc

        predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        return PredictionResponse(
            predicted_class=classes[predicted_index],
            confidence=round(probabilities[predicted_index] * 100, 2),
            model_version=request.app.state.model_version,
        )

    return application


app = create_app()
