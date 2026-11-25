"""Prepara paquetes versionados de un modelo existente sin cargarlo."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from src.manifest import ManifestValidationError, load_manifest, verify_artifact


_STABLE_VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")
_MANIFEST_FILENAME = "model-manifest.json"


def _check_version_compatibility(manifest: dict[str, Any], version: str) -> str:
    if not isinstance(version, str) or not _STABLE_VERSION.fullmatch(version):
        raise ManifestValidationError(
            "La versión debe usar el formato estable MAJOR.MINOR.PATCH, por ejemplo 1.0.0."
        )

    release_tag = f"model-v{version}"
    base_version = manifest["model"].get("version")
    if base_version is not None and base_version != version:
        raise ManifestValidationError(
            f"La versión solicitada '{version}' no coincide con model.version del manifiesto base ('{base_version}')."
        )

    base_tag = manifest["provenance"].get("release_tag")
    if base_tag is not None and base_tag != release_tag:
        raise ManifestValidationError(
            f"La etiqueta solicitada '{release_tag}' no coincide con provenance.release_tag del manifiesto base ('{base_tag}')."
        )
    return release_tag


def prepare_release_package(
    artifact_path: str | Path,
    manifest_path: str | Path,
    output_root: str | Path,
    version: str,
) -> Path:
    """Crea ``model-v<version>`` con el artefacto verificado y su manifiesto.

    El manifiesto de entrada y el artefacto original se leen, pero nunca se
    modifican. La carpeta final aparece únicamente después de que el paquete
    temporal haya pasado sus verificaciones de integridad.
    """
    manifest = load_manifest(manifest_path)
    release_tag = _check_version_compatibility(manifest, version)
    original_integrity = verify_artifact(manifest, artifact_path)

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / release_tag
    if destination.exists():
        raise ManifestValidationError(f"El destino ya existe y no se sobrescribirá: '{destination}'.")

    # mkdir es exclusivo y atómico. Las ejecuciones concurrentes de esta
    # función para la misma versión quedan serializadas por esta reserva.
    reservation = root / f".{release_tag}.lock"
    try:
        reservation.mkdir()
    except FileExistsError as exc:
        raise ManifestValidationError(
            f"La versión '{release_tag}' ya se está preparando en otro proceso."
        ) from exc

    artifact_source = Path(artifact_path)
    try:
        if destination.exists():
            raise ManifestValidationError(f"El destino ya existe: '{destination}'.")
        with tempfile.TemporaryDirectory(prefix=f".{release_tag}-", dir=root) as temporary_directory:
            staging = Path(temporary_directory) / release_tag
            staging.mkdir()
            artifact_copy = staging / manifest["model"]["artifact"]["filename"]
            shutil.copy2(artifact_source, artifact_copy)

            release_manifest = json.loads(json.dumps(manifest))
            release_manifest["model"]["version"] = version
            release_manifest["provenance"]["release_tag"] = release_tag
            manifest_copy = staging / _MANIFEST_FILENAME
            with manifest_copy.open("w", encoding="utf-8", newline="\n") as file:
                json.dump(release_manifest, file, ensure_ascii=False, indent=2)
                file.write("\n")

            copied_integrity = verify_artifact(release_manifest, artifact_copy)
            if copied_integrity != original_integrity:
                raise ManifestValidationError("La copia del artefacto no conserva la integridad del original.")

            # Detecta también una alteración concurrente del origen durante la copia.
            final_original_integrity = verify_artifact(manifest, artifact_source)
            if final_original_integrity != original_integrity:
                raise ManifestValidationError("El artefacto original cambió durante la preparación del paquete.")

            with manifest_copy.open("r", encoding="utf-8") as file:
                written_manifest = json.load(file)
            if written_manifest != release_manifest:
                raise ManifestValidationError("El manifiesto copiado no coincide con el manifiesto de Release esperado.")

            # rename dentro del mismo filesystem publica el directorio completo
            # de una sola vez y falla si el destino apareció durante el proceso.
            # La reserva exclusiva serializa a los preparadores de esta versión.
            staging.rename(destination)
    except FileExistsError as exc:
        raise ManifestValidationError(f"El destino ya existe y no se sobrescribirá: '{destination}'.") from exc

    finally:
        reservation.rmdir()

    return destination
