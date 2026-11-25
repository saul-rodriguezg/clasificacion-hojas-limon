"""Pruebas de registro MLflow con dobles; no requieren MLflow ni TensorFlow."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.manifest import ManifestValidationError, load_manifest
from src.track_model import (
    DEFAULT_TRACKING_URI,
    track_existing_model,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_MANIFEST_PATH = REPOSITORY_ROOT / "models" / "model-manifest.json"


class FakeRun:
    def __init__(self, run_id: str = "run-123") -> None:
        self.info = SimpleNamespace(run_id=run_id)
        self.exit_exception = None

    def __enter__(self) -> "FakeRun":
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        self.exit_exception = exception
        return False


class TrackExistingModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.artifact_path = self.root / "EfficientNetB0.keras"
        self.artifact_bytes = b"existing model artifact for MLflow tests"
        self.artifact_path.write_bytes(self.artifact_bytes)

        manifest = load_manifest(BASE_MANIFEST_PATH)
        manifest["model"]["artifact"]["size_bytes"] = len(self.artifact_bytes)
        manifest["model"]["artifact"]["sha256"] = hashlib.sha256(self.artifact_bytes).hexdigest()
        self.manifest_path = self.root / "model-manifest.json"
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.manifest = manifest

        self.fake_run = FakeRun()
        self.experiment = SimpleNamespace(experiment_id="exp-7", name="LemonGuard_AI")
        self.mlflow = MagicMock()
        self.mlflow.set_experiment.return_value = self.experiment
        self.mlflow.start_run.return_value = self.fake_run
        self.mlflow.get_tracking_uri.return_value = "sqlite:///mlflow.db"
        self.mlflow_context = patch.dict(sys.modules, {"mlflow": self.mlflow})
        self.mlflow_context.start()
        self.addCleanup(self.mlflow_context.stop)

    def _track(self, **kwargs):
        # El URI local relativo se usa desde el directorio temporal, fuera del repo.
        with patch("src.track_model.Path.cwd", return_value=self.root):
            return track_existing_model(
                self.artifact_path,
                self.manifest_path,
                **kwargs,
            )

    def test_logs_metrics_parameters_tags_and_both_artifacts(self) -> None:
        result = self._track(tracking_uri="sqlite:///mlflow.db")

        self.mlflow.log_metrics.assert_called_once_with(
            {name: float(value) for name, value in self.manifest["evaluation"]["metrics"].items()}
        )
        self.mlflow.log_params.assert_called_once_with(
            {
                "architecture": "EfficientNetB0",
                "keras_version": "3.13.2",
                "image_height": 224,
                "image_width": 224,
                "batch_size": 16,
                "epochs": 5,
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "loss": "sparse_categorical_crossentropy",
            }
        )
        tags = self.mlflow.set_tags.call_args.args[0]
        self.assertEqual(tags["model_architecture"], "EfficientNetB0")
        self.assertEqual(tags["evaluation_type"], "validation")
        self.assertEqual(tags["evaluation_status"], "preliminary")
        self.assertEqual(tags["evaluation_independent_test_set"], "false")
        self.assertEqual(tags["artifact_type"], "existing_model")
        self.assertEqual(tags["project"], "LemonGuard AI")
        self.assertEqual(tags["original_training_run_recorded"], "false")
        self.assertEqual(self.mlflow.log_artifact.call_count, 2)
        self.mlflow.log_artifact.assert_any_call(str(self.artifact_path), artifact_path="model")
        self.mlflow.log_artifact.assert_any_call(str(self.manifest_path), artifact_path="manifest")
        self.mlflow.start_run.assert_called_once_with(run_name="register_existing_model")
        self.assertEqual(result["run_id"], "run-123")
        self.assertEqual(result["experiment_id"], "exp-7")
        self.assertEqual(result["experiment_name"], "LemonGuard_AI")
        self.assertEqual(result["model_artifact_uri"], "runs:/run-123/model/EfficientNetB0.keras")

    def test_environment_variable_configures_tracking_uri(self) -> None:
        self.mlflow.get_tracking_uri.return_value = "http://tracking.example"
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://tracking.example"}):
            result = self._track()
        self.mlflow.set_tracking_uri.assert_called_once_with("http://tracking.example")
        self.assertEqual(result["tracking_uri"], "http://tracking.example")

    def test_default_uri_is_sqlite_mlflow_db(self) -> None:
        self.assertEqual(DEFAULT_TRACKING_URI, "sqlite:///mlflow.db")
        self._track()
        self.mlflow.set_tracking_uri.assert_called_once_with(DEFAULT_TRACKING_URI)

    def test_invalid_manifest_prevents_mlflow_run(self) -> None:
        invalid_manifest = copy.deepcopy(self.manifest)
        invalid_manifest["evaluation"]["status"] = "production"
        self.manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")

        with self.assertRaises(ManifestValidationError):
            self._track(tracking_uri="http://tracking.example")

        self.mlflow.set_experiment.assert_not_called()
        self.mlflow.start_run.assert_not_called()

    def test_artifact_integrity_failure_prevents_mlflow_run(self) -> None:
        self.artifact_path.write_bytes(b"tampered model artifact")

        with self.assertRaises(ManifestValidationError):
            self._track(tracking_uri="http://tracking.example")

        self.mlflow.set_experiment.assert_not_called()
        self.mlflow.start_run.assert_not_called()

    def test_mlflow_error_propagates_and_does_not_return_success(self) -> None:
        self.mlflow.log_artifact.side_effect = RuntimeError("tracking write failed")

        with self.assertRaisesRegex(RuntimeError, "tracking write failed"):
            self._track(tracking_uri="http://tracking.example")

        self.assertIsNotNone(self.fake_run.exit_exception)

    def test_local_tracking_is_rejected_from_inside_git_checkout(self) -> None:
        with patch("src.track_model.Path.cwd", return_value=REPOSITORY_ROOT):
            with self.assertRaisesRegex(ValueError, "dentro del repositorio"):
                track_existing_model(
                    self.artifact_path,
                    self.manifest_path,
                    tracking_uri=DEFAULT_TRACKING_URI,
                )
        self.mlflow.set_experiment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
