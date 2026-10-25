"""Validate and restore the exact companion archive using only the standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = ("adaptive-data-quality/data/", "adaptive-data-quality/results/v2/tool_exchange/")

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def safe_destination(root: Path, name: str) -> Path:
    p = PurePosixPath(name)
    if "\\" in name or p.is_absolute() or ".." in p.parts or ":" in name or not name.startswith(ALLOWED):
        raise ValueError(f"Unexpected asset path: {name!r}")
    destination = (root / Path(*p.parts)).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError("Asset resolves outside the repository.")
    return destination

def restore(archive: Path, root: Path, *, verify_only: bool = False, overwrite: bool = False) -> dict:
    manifest = json.loads((root / "assets_manifest.json").read_text(encoding="utf-8"))
    expected = {row["path"]: row for row in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise ValueError("Duplicate paths in trusted asset manifest.")
    for name in expected:
        safe_destination(root, name)
    with zipfile.ZipFile(archive) as z:
        names = [i.filename for i in z.infolist()]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members.")
        license_name = "LICENSES/Raha-Apache-2.0.txt"
        allowed_names = set(expected) | {"ASSET_MANIFEST.json", "ASSET_NOTICES.md", license_name}
        if set(names) != allowed_names:
            raise ValueError("Archive contents do not match the trusted manifest.")
        embedded = json.loads(z.read("ASSET_MANIFEST.json"))
        if embedded != manifest:
            raise ValueError("Embedded asset manifest does not match this source release.")
        if hashlib.sha256(z.read("ASSET_NOTICES.md")).hexdigest() != manifest["notices_sha256"]:
            raise ValueError("Companion notices differ from the trusted manifest.")
        if z.read(license_name) != (root / "third_party" / "raha" / "LICENSE.upstream.md").read_bytes():
            raise ValueError("Companion upstream license differs from the source release.")
        root.joinpath(".artifacts").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="restore-", dir=root / ".artifacts") as temporary:
            staged = Path(temporary)
            to_install = []
            reused = 0
            total = 0
            for name, record in sorted(expected.items()):
                info = z.getinfo(name)
                if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f"Unsupported archive member: {name}")
                if info.file_size != record["bytes"]:
                    raise ValueError(f"Size differs from trusted manifest: {name}")
                destination = safe_destination(root, name)
                if destination.exists() and not overwrite and digest(destination) != record["sha256"]:
                    raise FileExistsError(f"Conflicting file: {name}. Use --overwrite only to replace it with the frozen snapshot.")
                output = None
                target = staged / name
                h = hashlib.sha256()
                count = 0
                try:
                    if not verify_only:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        output = target.open("wb")
                    with z.open(info) as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            count += len(block)
                            if count > record["bytes"]:
                                raise ValueError("Archive expands beyond declared size.")
                            h.update(block)
                            if output:
                                output.write(block)
                finally:
                    if output:
                        output.close()
                if count != record["bytes"] or h.hexdigest() != record["sha256"]:
                    raise ValueError(f"Content hash differs from trusted manifest: {name}")
                total += count
                if not verify_only:
                    if destination.exists() and digest(destination) == record["sha256"]:
                        reused += 1
                    else:
                        to_install.append((target, destination))
            # Verify every member before installing any file.
            for source, destination in to_install:
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.replace(destination)
    return {"verified_files": len(expected), "verified_bytes": total,
            "installed_files": len(to_install), "reused_files": reused, "verify_only": verify_only}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", type=Path)
    ap.add_argument("--verify-only", action="store_true", help="Check every compressed member without installing data.")
    ap.add_argument("--overwrite", action="store_true", help="Replace conflicting input files only after complete verification.")
    args = ap.parse_args()
    print(json.dumps(restore(args.archive.resolve(), ROOT, verify_only=args.verify_only, overwrite=args.overwrite), indent=2))

if __name__ == "__main__":
    main()
