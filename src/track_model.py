"""Registra un artefacto de modelo existente en MLflow, sin cargarlo."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.manifest import load_manifest, verify_artifact


DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "LemonGuard_AI"
RUN_NAME = "register_existing_model"


def _repository_root(path: Path) -> Path | None:
    current = path.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _ensure_local_store_outside_repository(tracking_uri: str) -> None:
    """Evita que el backend o los artefactos locales acaben dentro del repo.

    MLflow crea por defecto la base SQLite y su directorio de artefactos en
    relación con el directorio de trabajo. Por ello, con un backend local el
    llamador debe ejecutar desde fuera del checkout. Los backends remotos no
    escriben esos datos en el checkout local.
    """
    if not (tracking_uri.startswith("sqlite:") or tracking_uri.startswith("file:")):
        return

    cwd = Path.cwd().resolve()
    repository = _repository_root(cwd)
    if repository is not None:
        raise ValueError(
            "El tracking local de MLflow no se ejecutará desde dentro del repositorio "
            "para evitar añadir mlflow.db o sus artefactos al árbol Git. Ejecuta desde "
            "un directorio externo al checkout o configura un tracking URI remoto."
        )


def track_existing_model(
    artifact_path: str | Path,
    manifest_path: str | Path,
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> dict[str, str]:
    """Registra el archivo `.keras` existente y sus datos en una ejecución MLflow.

    La función valida los bytes antes de importar MLflow. No carga ni
    reserializa el modelo, y el run describe una incorporación de artefacto,
    no la ejecución de entrenamiento original.

    Devuelve identificadores y URI para abrir el experimento y recuperar los
    artefactos registrados. Los errores de validación o de MLflow se propagan;
    no se devuelve un resultado de éxito si el run falla.
    """
    manifest = load_manifest(manifest_path)
    integrity = verify_artifact(manifest, artifact_path)

    effective_tracking_uri = (
        tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    )
    if not effective_tracking_uri.strip():
        raise ValueError("El tracking URI de MLflow no puede estar vacío.")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("El nombre del experimento no puede estar vacío.")
    _ensure_local_store_outside_repository(effective_tracking_uri)

    # Importación diferida: CI y quienes solo validan el manifiesto no necesitan
    # instalar MLflow, TensorFlow, Keras ni KerasHub.
    import mlflow

    artifact = Path(artifact_path)
    manifest_file = Path(manifest_path)
    model = manifest["model"]
    evaluation = manifest["evaluation"]
    input_shape = model["input"]["shape"]

    params: dict[str, Any] = {
        "architecture": model["architecture"],
        "keras_version": model["saved_with"]["keras"],
        "image_height": input_shape[1],
        "image_width": input_shape[2],
        "batch_size": 16,
        "epochs": 5,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "loss": "sparse_categorical_crossentropy",
    }
    metrics = {
        name: float(value)
        for name, value in evaluation["metrics"].items()
    }
    tags = {
        "model_architecture": model["architecture"],
        "evaluation_type": evaluation["split"],
        "evaluation_status": evaluation["status"],
        "evaluation_independent_test_set": str(evaluation["independent_test_set"]).lower(),
        "evaluation_sample_count": str(evaluation["sample_count"]),
        "artifact_type": "existing_model",
        "artifact_sha256": integrity["sha256"],
        "artifact_size_bytes": str(integrity["size_bytes"]),
        "project": "LemonGuard AI",
        "registration_type": "existing_artifact_import",
        "original_training_run_recorded": "false",
    }

    mlflow.set_tracking_uri(effective_tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        mlflow.log_artifact(str(artifact), artifact_path="model")
        mlflow.log_artifact(str(manifest_file), artifact_path="manifest")

    run_id = run.info.run_id
    return {
        "run_id": run_id,
        "experiment_id": str(experiment.experiment_id),
        "experiment_name": experiment.name,
        "tracking_uri": mlflow.get_tracking_uri(),
        "run_uri": f"runs:/{run_id}",
        "model_artifact_uri": f"runs:/{run_id}/model/{artifact.name}",
        "manifest_artifact_uri": f"runs:/{run_id}/manifest/{manifest_file.name}",
    }
