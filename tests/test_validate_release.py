"""Pruebas de validación de GitHub Release sin dependencias externas."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.manifest import ManifestValidationError, load_manifest
from src.validate_release import validate_release_package


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_MANIFEST_PATH = REPOSITORY_ROOT / "models" / "model-manifest.json"


class ValidateReleasePackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.root = Path(self.temp_directory.name)
        self.artifact_path = self.root / "EfficientNetB0.keras"
        self.artifact_bytes = b"temporary model bytes for Release validation"
        self.artifact_path.write_bytes(self.artifact_bytes)

        self.manifest = copy.deepcopy(load_manifest(BASE_MANIFEST_PATH))
        self.manifest["model"]["version"] = "1.0.0"
        self.manifest["model"]["artifact"]["size_bytes"] = len(self.artifact_bytes)
        self.manifest["model"]["artifact"]["sha256"] = hashlib.sha256(self.artifact_bytes).hexdigest()
        self.manifest["provenance"]["release_tag"] = "model-v1.0.0"
        self.manifest_path = self.root / "model-manifest.json"
        self._write_manifest(self.manifest)

    def _write_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _validate(self, release_tag: str = "model-v1.0.0") -> dict[str, str | int]:
        return validate_release_package(self.artifact_path, self.manifest_path, release_tag)

    def test_accepts_valid_release_package(self) -> None:
        result = self._validate()
        self.assertEqual(result["version"], "1.0.0")
        self.assertEqual(result["release_tag"], "model-v1.0.0")
        self.assertEqual(result["artifact_filename"], "EfficientNetB0.keras")
        self.assertEqual(result["size_bytes"], len(self.artifact_bytes))
        self.assertEqual(result["sha256"], hashlib.sha256(self.artifact_bytes).hexdigest())

    def test_rejects_invalid_release_tag_format(self) -> None:
        for tag in ("v1.0.0", "model-v1.0", "model-v01.0.0", "model-v1.0.0-rc.1", "model-v1.0.0/"):
            with self.subTest(tag=tag), self.assertRaises(ManifestValidationError):
                self._validate(tag)

    def test_rejects_version_mismatch(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["model"]["version"] = "1.0.1"
        self._write_manifest(manifest)
        with self.assertRaisesRegex(ManifestValidationError, "no coincide con el tag"):
            self._validate()

    def test_rejects_provenance_release_tag_mismatch(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["provenance"]["release_tag"] = "model-v1.0.1"
        self._write_manifest(manifest)
        with self.assertRaisesRegex(ManifestValidationError, "provenance.release_tag"):
            self._validate()

    def test_rejects_incorrect_sha256(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["model"]["artifact"]["sha256"] = "0" * 64
        self._write_manifest(manifest)
        with self.assertRaisesRegex(ManifestValidationError, "SHA-256"):
            self._validate()

    def test_rejects_incorrect_artifact_size(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["model"]["artifact"]["size_bytes"] += 1
        self._write_manifest(manifest)
        with self.assertRaisesRegex(ManifestValidationError, "tamaño"):
            self._validate()

    def test_rejects_missing_artifact(self) -> None:
        self.artifact_path.unlink()
        with self.assertRaisesRegex(ManifestValidationError, "No existe un archivo"):
            self._validate()


if __name__ == "__main__":
    unittest.main()
