"""Pruebas HTTP para la API sin cargar TensorFlow ni el artefacto real."""

from __future__ import annotations

import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


_API_TEST_DEPENDENCIES = ("fastapi", "httpx", "multipart", "PIL")
API_TESTS_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in _API_TEST_DEPENDENCIES)

if API_TESTS_AVAILABLE:
    from fastapi.testclient import TestClient
    from PIL import Image

    from src.api import DEFAULT_CLASS_NAMES, create_app


class FakePredictor:
    def __init__(self, probabilities: list[float] | None = None, error: Exception | None = None) -> None:
        self.probabilities = probabilities or [0.8, 0.2]
        self.error = error
        self.images: list[bytes] = []

    def predict(self, image_bytes: bytes) -> list[float]:
        self.images.append(image_bytes)
        if self.error is not None:
            raise self.error
        return self.probabilities


@unittest.skipUnless(
    API_TESTS_AVAILABLE,
    "Instala las dependencias HTTP de requirements-api.txt para ejecutar las pruebas de API.",
)
class InferenceAPITests(unittest.TestCase):
    @staticmethod
    def _valid_png() -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), color=(80, 150, 35)).save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _client(predictor: FakePredictor):
        app = create_app(
            predictor=predictor,
            model_version="1.0.0",
            class_names=DEFAULT_CLASS_NAMES,
        )
        return TestClient(app)

    def test_health_check_reports_loaded_model_and_version(self) -> None:
        with self._client(FakePredictor()) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "ok", "model_loaded": True, "model_version": "1.0.0"},
        )

    def test_predicts_both_classes(self) -> None:
        for probabilities, expected_class in (
            ([0.91, 0.09], "diseased"),
            ([0.14, 0.86], "healthy"),
        ):
            predictor = FakePredictor(probabilities=probabilities)
            with self.subTest(expected_class=expected_class), self._client(predictor) as client:
                response = client.post(
                    "/predict",
                    files={"image": ("leaf.png", self._valid_png(), "image/png")},
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["predicted_class"], expected_class)
            self.assertEqual(response.json()["model_version"], "1.0.0")
            self.assertEqual(len(predictor.images), 1)

    def test_rejects_invalid_image(self) -> None:
        predictor = FakePredictor()
        with self._client(predictor) as client:
            response = client.post(
                "/predict",
                files={"image": ("not-an-image.bin", b"not an image", "application/octet-stream")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(predictor.images), 0)

    def test_returns_http_500_when_inference_fails(self) -> None:
        predictor = FakePredictor(error=RuntimeError("simulated inference failure"))
        with self._client(predictor) as client:
            with self.assertLogs("lemonguard.api", level="ERROR"):
                response = client.post(
                    "/predict",
                    files={"image": ("leaf.png", self._valid_png(), "image/png")},
                )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "No fue posible completar la inferencia.")

    def test_response_shape_and_confidence_range(self) -> None:
        with self._client(FakePredictor(probabilities=[0.73, 0.27])) as client:
            response = client.post(
                "/predict",
                files={"image": ("leaf.png", self._valid_png(), "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(set(payload), {"predicted_class", "confidence", "model_version"})
        self.assertGreaterEqual(payload["confidence"], 0)
        self.assertLessEqual(payload["confidence"], 100)

    def test_model_loader_runs_once_for_multiple_requests(self) -> None:
        fake_predictor = FakePredictor()
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "EfficientNetB0.keras"
            app = create_app(
                model_path=model_path,
                model_version="1.0.0",
                class_names=DEFAULT_CLASS_NAMES,
            )
            with patch("src.api.KerasPredictor", return_value=fake_predictor) as loader:
                with TestClient(app) as client:
                    client.get("/health")
                    client.post(
                        "/predict",
                        files={"image": ("leaf.png", self._valid_png(), "image/png")},
                    )
                    client.post(
                        "/predict",
                        files={"image": ("leaf.png", self._valid_png(), "image/png")},
                    )

                loader.assert_called_once_with(model_path)


if __name__ == "__main__":
    unittest.main()
