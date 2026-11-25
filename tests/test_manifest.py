"""Pruebas unitarias del manifiesto que no requieren TensorFlow."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.manifest import (
    ManifestValidationError,
    load_manifest,
    validate_manifest,
    verify_artifact,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "models" / "model-manifest.json"


class ManifestValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest(MANIFEST_PATH)

    def test_repository_manifest_is_valid(self) -> None:
        validate_manifest(self.manifest)

    def test_unknown_provenance_fields_remain_null(self) -> None:
        provenance = self.manifest["provenance"]
        self.assertTrue(all(value is None for value in provenance.values()))

    def test_model_integrity_values_match_recorded_original(self) -> None:
        artifact = self.manifest["model"]["artifact"]
        self.assertEqual(artifact["filename"], "EfficientNetB0.keras")
        self.assertEqual(artifact["size_bytes"], 16_855_030)
        self.assertEqual(
            artifact["sha256"],
            "a359031a8ea0c08f9a28e1e10b5ee8a5669958c40d9504845a8cf7d5667606ac",
        )

    def test_rejects_invalid_sha256(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["model"]["artifact"]["sha256"] = "not-a-sha256"
        with self.assertRaises(ManifestValidationError):
            validate_manifest(manifest)

    def test_rejects_class_support_that_does_not_match_sample_count(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["evaluation"]["class_support"]["healthy"] -= 1
        with self.assertRaises(ManifestValidationError):
            validate_manifest(manifest)

    def test_rejects_metric_outside_unit_interval(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["evaluation"]["metrics"]["diseased_recall"] = 1.2
        with self.assertRaises(ManifestValidationError):
            validate_manifest(manifest)

    def test_rejects_output_class_dimension_mismatch(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["model"]["output"]["shape"][1] = 3
        with self.assertRaises(ManifestValidationError):
            validate_manifest(manifest)

    def test_rejects_decimal_output_class_dimension(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["model"]["output"]["shape"][1] = 2.0
        with self.assertRaises(ManifestValidationError):
            validate_manifest(manifest)

    def test_rejects_decimal_class_index(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["model"]["output"]["classes"][0]["index"] = 0.0
        with self.assertRaises(ManifestValidationError):
            validate_manifest(manifest)

    def test_verifies_external_artifact_by_size_and_sha256(self) -> None:
        content = b"temporary model bytes for checksum test"
        manifest = copy.deepcopy(self.manifest)
        manifest["model"]["artifact"]["size_bytes"] = len(content)
        manifest["model"]["artifact"]["sha256"] = hashlib.sha256(content).hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "EfficientNetB0.keras"
            artifact_path.write_bytes(content)
            result = verify_artifact(manifest, artifact_path)

        self.assertEqual(result["size_bytes"], len(content))
        self.assertEqual(result["sha256"], hashlib.sha256(content).hexdigest())

    def test_rejects_external_artifact_with_wrong_hash(self) -> None:
        content = b"temporary model bytes for checksum test"
        manifest = copy.deepcopy(self.manifest)
        manifest["model"]["artifact"]["size_bytes"] = len(content)
        manifest["model"]["artifact"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "EfficientNetB0.keras"
            artifact_path.write_bytes(content)
            with self.assertRaises(ManifestValidationError):
                verify_artifact(manifest, artifact_path)

    def test_rejects_external_artifact_with_wrong_size(self) -> None:
        content = b"temporary model bytes for checksum test"
        manifest = copy.deepcopy(self.manifest)
        manifest["model"]["artifact"]["sha256"] = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "EfficientNetB0.keras"
            artifact_path.write_bytes(content)
            with self.assertRaises(ManifestValidationError):
                verify_artifact(manifest, artifact_path)

    def test_manifest_is_valid_json(self) -> None:
        with MANIFEST_PATH.open("r", encoding="utf-8") as file:
            parsed = json.load(file)
        self.assertEqual(parsed, self.manifest)


if __name__ == "__main__":
    unittest.main()
