"""Security-relevant restoration checks; no research data or services are needed."""
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
import warnings
import zipfile

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("asset_restore", REPO / "release_tools" / "restore_assets.py")
restore_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore_module)
NAME = "adaptive-data-quality/data/original/sample.txt"
NOTICE = b"Test notice"
LICENSE = b"Test upstream notice"

class RestoreTests(unittest.TestCase):
    def setUp(self):
        (REPO / ".artifacts").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="asset-test-", dir=REPO / ".artifacts")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "assets.zip"

    def prepare(self, payload=b"frozen", *, member=None, duplicate=False, extra=None, symlink=False):
        manifest = {"schema_version": 1, "notices_sha256": hashlib.sha256(NOTICE).hexdigest(),
                    "files": [{"path": NAME, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]}
        (self.root / "assets_manifest.json").write_text(json.dumps(manifest))
        upstream = self.root / "third_party" / "raha" / "LICENSE.upstream.md"
        upstream.parent.mkdir(parents=True)
        upstream.write_bytes(LICENSE)
        with zipfile.ZipFile(self.archive, "w") as z:
            info = zipfile.ZipInfo(NAME)
            if symlink:
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, payload if member is None else member)
            z.writestr("ASSET_MANIFEST.json", json.dumps(manifest))
            z.writestr("ASSET_NOTICES.md", NOTICE)
            z.writestr("LICENSES/Raha-Apache-2.0.txt", LICENSE)
            if extra:
                z.writestr(extra, b"unexpected")
            if duplicate:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    z.writestr(NAME, payload)

    def test_valid_restore_and_reuse(self):
        self.prepare()
        result = restore_module.restore(self.archive, self.root)
        self.assertEqual(result["installed_files"], 1)
        self.assertEqual((self.root / NAME).read_bytes(), b"frozen")
        second = restore_module.restore(self.archive, self.root)
        self.assertEqual(second["reused_files"], 1)

    def test_verify_only_does_not_install(self):
        self.prepare()
        result = restore_module.restore(self.archive, self.root, verify_only=True)
        self.assertEqual(result["verified_files"], 1)
        self.assertFalse((self.root / NAME).exists())

    def test_corrupt_content_never_installs(self):
        self.prepare(member=b"broken")
        with self.assertRaises(ValueError):
            restore_module.restore(self.archive, self.root)
        self.assertFalse((self.root / NAME).exists())

    def test_unexpected_traversal_rejected(self):
        self.prepare(extra="../outside.txt")
        with self.assertRaises(ValueError):
            restore_module.restore(self.archive, self.root)
        self.assertFalse((self.root / NAME).exists())
        self.assertFalse((self.root.parent / "outside.txt").exists())

    def test_duplicate_rejected(self):
        self.prepare(duplicate=True)
        with self.assertRaises(ValueError):
            restore_module.restore(self.archive, self.root)

    def test_symlink_rejected(self):
        self.prepare(symlink=True)
        with self.assertRaises(ValueError):
            restore_module.restore(self.archive, self.root)

    def test_conflict_keeps_existing_data(self):
        self.prepare()
        target = self.root / NAME
        target.parent.mkdir(parents=True)
        target.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            restore_module.restore(self.archive, self.root)
        self.assertEqual(target.read_bytes(), b"existing")

if __name__ == "__main__":
    unittest.main()
