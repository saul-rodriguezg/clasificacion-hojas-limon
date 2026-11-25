"""Carga, valida y comprueba la integridad de manifiestos de modelo.

Este módulo usa únicamente la biblioteca estándar de Python para que pueda
ejecutarse en CI sin instalar TensorFlow ni cargar el modelo.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ManifestValidationError(ValueError):
    """El manifiesto no cumple el contrato esperado."""


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"'{field}' debe ser un objeto JSON.")
    return value


def _require_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"'{field}' debe ser una cadena no vacía.")
    return value


def _require_positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestValidationError(f"'{field}' debe ser un entero positivo.")
    return value


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Valida la estructura y las invariantes del manifiesto.

    Los campos de procedencia admiten ``null`` mientras no se conozca su valor.
    Lanza :class:`ManifestValidationError` ante cualquier inconsistencia.
    """
    root = _require_mapping(manifest, "manifest")
    if root.get("schema_version") != "1.0":
        raise ManifestValidationError("'schema_version' debe ser '1.0'.")

    model = _require_mapping(root.get("model"), "model")
    for field in ("name", "framework", "format", "architecture"):
        _require_non_empty_string(model.get(field), f"model.{field}")

    version = model.get("version")
    if version is not None and (not isinstance(version, str) or not version.strip()):
        raise ManifestValidationError("'model.version' debe ser null o una cadena no vacía.")

    artifact = _require_mapping(model.get("artifact"), "model.artifact")
    _require_non_empty_string(artifact.get("filename"), "model.artifact.filename")
    _require_positive_integer(artifact.get("size_bytes"), "model.artifact.size_bytes")
    sha256 = artifact.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ManifestValidationError("'model.artifact.sha256' debe contener 64 caracteres hexadecimales.")
    try:
        int(sha256, 16)
    except ValueError as exc:
        raise ManifestValidationError("'model.artifact.sha256' debe contener 64 caracteres hexadecimales.") from exc

    saved_with = _require_mapping(model.get("saved_with"), "model.saved_with")
    _require_non_empty_string(saved_with.get("keras"), "model.saved_with.keras")

    input_spec = _require_mapping(model.get("input"), "model.input")
    input_shape = input_spec.get("shape")
    if (
        not isinstance(input_shape, list)
        or len(input_shape) != 4
        or input_shape[0] is not None
        or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in input_shape[1:])
    ):
        raise ManifestValidationError("'model.input.shape' debe ser [null, alto, ancho, canales].")

    output_spec = _require_mapping(model.get("output"), "model.output")
    output_shape = output_spec.get("shape")
    classes = output_spec.get("classes")
    if not isinstance(classes, list) or len(classes) < 2:
        raise ManifestValidationError("'model.output.classes' debe contener al menos dos clases.")
    class_names: list[str] = []
    for expected_index, item in enumerate(classes):
        class_item = _require_mapping(item, f"model.output.classes[{expected_index}]")
        index = class_item.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index != expected_index:
            raise ManifestValidationError("Los índices de clase deben ser consecutivos y comenzar en 0.")
        class_names.append(_require_non_empty_string(class_item.get("name"), "class.name"))
    if len(set(class_names)) != len(class_names):
        raise ManifestValidationError("Los nombres de clase deben ser únicos.")
    if (
        not isinstance(output_shape, list)
        or len(output_shape) != 2
        or output_shape[0] is not None
        or isinstance(output_shape[1], bool)
        or not isinstance(output_shape[1], int)
        or output_shape[1] != len(classes)
    ):
        raise ManifestValidationError("'model.output.shape' debe concordar con el número de clases.")
    if output_spec.get("activation") != "softmax":
        raise ManifestValidationError("'model.output.activation' debe ser 'softmax'.")

    preprocessing = _require_mapping(model.get("preprocessing"), "model.preprocessing")
    if not isinstance(preprocessing.get("integrated"), bool):
        raise ManifestValidationError("'model.preprocessing.integrated' debe ser booleano.")
    _require_non_empty_string(preprocessing.get("implementation"), "model.preprocessing.implementation")

    evaluation = _require_mapping(root.get("evaluation"), "evaluation")
    if evaluation.get("status") != "preliminary" or evaluation.get("split") != "validation":
        raise ManifestValidationError("La evaluación debe identificarse como preliminar y de validación.")
    sample_count = _require_positive_integer(evaluation.get("sample_count"), "evaluation.sample_count")
    if evaluation.get("independent_test_set") is not False:
        raise ManifestValidationError("'evaluation.independent_test_set' debe ser false.")

    class_support = _require_mapping(evaluation.get("class_support"), "evaluation.class_support")
    support_total = 0
    for name in class_names:
        support_total += _require_positive_integer(class_support.get(name), f"evaluation.class_support.{name}")
    if support_total != sample_count:
        raise ManifestValidationError("La suma de class_support debe coincidir con sample_count.")

    metrics = _require_mapping(evaluation.get("metrics"), "evaluation.metrics")
    rate_metrics = (
        "accuracy",
        "diseased_precision",
        "diseased_recall",
        "diseased_f1",
        "healthy_precision",
        "healthy_recall",
        "healthy_f1",
    )
    for name in rate_metrics:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ManifestValidationError(f"'evaluation.metrics.{name}' debe estar entre 0 y 1.")
    for name in ("average_latency_ms_per_image", "latency_stddev_ms_per_image"):
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ManifestValidationError(f"'evaluation.metrics.{name}' debe ser un número finito no negativo.")

    provenance = _require_mapping(root.get("provenance"), "provenance")
    for field in (
        "training_run_id",
        "dataset_name",
        "dataset_version",
        "dataset_sha256",
        "source_commit",
        "release_tag",
        "created_at",
    ):
        value = provenance.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ManifestValidationError(f"'provenance.{field}' debe ser null o una cadena no vacía.")


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Lee un manifiesto JSON y valida su contenido."""
    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"No se pudo leer el manifiesto '{manifest_path}': {exc}") from exc
    validate_manifest(manifest)
    return manifest


def verify_artifact(manifest: Mapping[str, Any], artifact_path: str | Path) -> dict[str, Any]:
    """Comprueba el nombre, tamaño y SHA-256 del artefacto contra el manifiesto.

    El artefacto puede estar fuera del repositorio. El cálculo se hace en bloques,
    así que no es necesario cargar el archivo completo en memoria.
    """
    validate_manifest(manifest)
    expected = manifest["model"]["artifact"]
    path = Path(artifact_path)
    if not path.is_file():
        raise ManifestValidationError(f"No existe un archivo de modelo en '{path}'.")
    if path.name != expected["filename"]:
        raise ManifestValidationError(
            f"El nombre del artefacto es '{path.name}', se esperaba '{expected['filename']}'."
        )

    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            size_bytes += len(chunk)
            digest.update(chunk)

    actual_sha256 = digest.hexdigest()
    if size_bytes != expected["size_bytes"]:
        raise ManifestValidationError(
            f"El tamaño del artefacto es {size_bytes} bytes; se esperaba {expected['size_bytes']}."
        )
    if actual_sha256 != expected["sha256"].lower():
        raise ManifestValidationError("El SHA-256 del artefacto no coincide con el manifiesto.")

    return {"filename": path.name, "size_bytes": size_bytes, "sha256": actual_sha256}
