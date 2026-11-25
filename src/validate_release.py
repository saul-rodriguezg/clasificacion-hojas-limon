"""Valida un paquete de modelo descargado de una GitHub Release."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from src.manifest import ManifestValidationError, load_manifest, verify_artifact


_RELEASE_TAG = re.compile(
    r"model-v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z"
)


def validate_release_package(
    artifact_path: str | Path,
    manifest_path: str | Path,
    release_tag: str,
) -> dict[str, str | int]:
    """Valida la versión/tag del manifiesto y la integridad del artefacto."""
    tag_match = _RELEASE_TAG.fullmatch(release_tag) if isinstance(release_tag, str) else None
    if tag_match is None:
        raise ManifestValidationError(
            "La etiqueta debe tener formato model-vMAJOR.MINOR.PATCH, por ejemplo model-v1.0.0."
        )

    version = ".".join(tag_match.groups())
    manifest = load_manifest(manifest_path)
    model_version = manifest["model"].get("version")
    if model_version != version:
        raise ManifestValidationError(
            f"La versión del manifiesto ('{model_version}') no coincide con el tag ('{version}')."
        )

    manifest_tag = manifest["provenance"].get("release_tag")
    if manifest_tag != release_tag:
        raise ManifestValidationError(
            f"provenance.release_tag ('{manifest_tag}') no coincide con '{release_tag}'."
        )

    integrity = verify_artifact(manifest, artifact_path)
    return {
        "version": version,
        "release_tag": release_tag,
        "artifact_filename": integrity["filename"],
        "size_bytes": integrity["size_bytes"],
        "sha256": integrity["sha256"],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="Ruta al archivo EfficientNetB0.keras.")
    parser.add_argument("--manifest", required=True, help="Ruta al model-manifest.json de la Release.")
    parser.add_argument("--release-tag", required=True, help="Tag esperado, por ejemplo model-v1.0.0.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = validate_release_package(args.artifact, args.manifest, args.release_tag)
    except (ManifestValidationError, OSError, ValueError) as exc:
        print(f"Validación de Release fallida: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
