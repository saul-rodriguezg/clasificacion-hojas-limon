"""Pruebas del empaquetado versionado sin dependencias de ML."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from src.manifest import ManifestValidationError, load_manifest
from src.model_release import prepare_release_package


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_MANIFEST_PATH = REPOSITORY_ROOT / "models" / "model-manifest.json"


class PrepareReleasePackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_content = b"temporary EfficientNetB0 test artifact"
        self.original_hash = hashlib.sha256(self.original_content).hexdigest()
        self.manifest = load_manifest(BASE_MANIFEST_PATH)
        self.manifest["model"]["artifact"]["size_bytes"] = len(self.original_content)
        self.manifest["model"]["artifact"]["sha256"] = self.original_hash

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.artifact_path = self.root / "EfficientNetB0.keras"
        self.artifact_path.write_bytes(self.original_content)
        self.manifest_path = self.root / "base-manifest.json"
        self._write_manifest(self.manifest)
        self.base_manifest_bytes = self.manifest_path.read_bytes()
        self.original_artifact_bytes = self.artifact_path.read_bytes()
        self.output_root = self.root / "packages"

    def _write_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _prepare(self, version: str = "1.0.0") -> Path:
        return prepare_release_package(
            self.artifact_path,
            self.manifest_path,
            self.output_root,
            version,
        )

    def test_generates_package_with_expected_files(self) -> None:
        package = self._prepare()
        self.assertEqual(package.name, "model-v1.0.0")
        self.assertEqual(
            {path.name for path in package.iterdir()},
            {"EfficientNetB0.keras", "model-manifest.json"},
        )

    def test_version_and_release_tag_match(self) -> None:
        package = self._prepare("2.3.4")
        release_manifest = json.loads((package / "model-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(release_manifest["model"]["version"], "2.3.4")
        self.assertEqual(release_manifest["provenance"]["release_tag"], "model-v2.3.4")

    def test_artifact_copy_preserves_size_and_hash(self) -> None:
        package = self._prepare()
        copied = package / "EfficientNetB0.keras"
        self.assertEqual(copied.stat().st_size, len(self.original_content))
        self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), self.original_hash)

    def test_original_artifact_and_base_manifest_remain_unchanged(self) -> None:
        self._prepare()
        self.assertEqual(self.artifact_path.read_bytes(), self.original_artifact_bytes)
        self.assertEqual(self.manifest_path.read_bytes(), self.base_manifest_bytes)

    def test_unknown_provenance_fields_remain_null(self) -> None:
        package = self._prepare()
        provenance = json.loads((package / "model-manifest.json").read_text(encoding="utf-8"))["provenance"]
        self.assertIsNone(provenance["training_run_id"])
        self.assertIsNone(provenance["dataset_name"])
        self.assertIsNone(provenance["dataset_version"])
        self.assertIsNone(provenance["dataset_sha256"])
        self.assertIsNone(provenance["source_commit"])
        self.assertIsNone(provenance["created_at"])

    def test_rejects_invalid_versions(self) -> None:
        for version in ("v1.0.0", "1.0", "1.0.0-rc.1", "01.0.0", "1.0.0+build", "1.0.0 "):
            with self.subTest(version=version), self.assertRaises(ManifestValidationError):
                self._prepare(version)

    def test_rejects_incorrect_hash(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["model"]["artifact"]["sha256"] = "0" * 64
        self._write_manifest(manifest)
        with self.assertRaises(ManifestValidationError):
            self._prepare()
        self.assertFalse(self.output_root.exists())

    def test_rejects_incorrect_size(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["model"]["artifact"]["size_bytes"] += 1
        self._write_manifest(manifest)
        with self.assertRaises(ManifestValidationError):
            self._prepare()
        self.assertFalse(self.output_root.exists())

    def test_rejects_existing_destination_without_overwriting(self) -> None:
        self.output_root.mkdir()
        destination = self.output_root / "model-v1.0.0"
        destination.mkdir()
        sentinel = destination / "keep.txt"
        sentinel.write_text("published package", encoding="utf-8")
        with self.assertRaises(ManifestValidationError):
            self._prepare()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "published package")

    def test_rejects_version_reserved_by_concurrent_process(self) -> None:
        self.output_root.mkdir()
        reservation = self.output_root / ".model-v1.0.0.lock"
        reservation.mkdir()
        with self.assertRaisesRegex(ManifestValidationError, "otro proceso"):
            self._prepare()
        self.assertTrue(reservation.is_dir())
        self.assertFalse((self.output_root / "model-v1.0.0").exists())

    def test_cleans_temporary_directory_and_reservation_after_error(self) -> None:
        with patch("src.model_release.shutil.copy2", side_effect=OSError("simulated copy failure")):
            with self.assertRaisesRegex(OSError, "simulated copy failure"):
                self._prepare()
        self.assertEqual(list(self.output_root.iterdir()), [])

    def test_rejects_version_incompatible_with_base_manifest(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["model"]["version"] = "1.2.3"
        self._write_manifest(manifest)
        with self.assertRaises(ManifestValidationError):
            self._prepare("1.0.0")


if __name__ == "__main__":
    unittest.main()
