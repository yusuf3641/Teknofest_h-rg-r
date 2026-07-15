from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_sources(config: dict[str, Any], names: set[str] | None) -> list[dict[str, Any]]:
    sources = config["sources"]
    if names is None:
        return sources
    known = {source["name"] for source in sources}
    unknown = names - known
    if unknown:
        raise SystemExit(f"unknown source(s): {', '.join(sorted(unknown))}")
    return [source for source in sources if source["name"] in names]


def write_source_manifest(source: dict[str, Any], destination: Path, status: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": source["name"],
        "kind": source["kind"],
        "url": source.get("url") or source.get("repo_id"),
        "license": source.get("license", "unknown"),
        "modality": source.get("modality", "unknown"),
        "notes": source.get("notes", ""),
        "status": status,
    }
    (destination / "hurgor_source_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def download_git(source: dict[str, Any], *, update: bool) -> None:
    destination = Path(source["destination"])
    if (destination / ".git").is_dir():
        if update:
            run(["git", "pull", "--ff-only"], cwd=destination)
        write_source_manifest(source, destination, "present")
        return
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"destination is not empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", source["url"], str(destination)])
    write_source_manifest(source, destination, "downloaded")


def download_huggingface_file(source: dict[str, Any], *, force: bool) -> None:
    destination = Path(source["destination"])
    if destination.is_file() and not force:
        write_source_manifest(source, destination.parent, "present")
        return
    repo_id = source["repo_id"]
    filename = source["filename"]
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)
    write_source_manifest(source, destination.parent, "downloaded")


def google_drive_warning_url(file_id: str) -> str:
    query = urllib.parse.urlencode({"export": "download", "id": file_id})
    return f"https://drive.google.com/uc?{query}"


def download_google_drive_file(source: dict[str, Any], *, force: bool, allow_large: bool) -> None:
    destination = Path(source["destination"])
    expected_size_gb = float(source.get("expected_size_gb", 0))
    if destination.is_file() and not force:
        write_source_manifest(source, destination.parent, "present")
        return
    if expected_size_gb >= 20 and not allow_large:
        raise SystemExit(
            f"{source['name']} is declared as {expected_size_gb:g} GB. "
            "Pass --allow-large-downloads only after confirming disk space."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = google_drive_warning_url(source["file_id"])
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if temporary.stat().st_size < 1024 * 1024:
        preview = temporary.read_text(encoding="utf-8", errors="ignore")[:500]
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            "Google Drive returned a warning/login page instead of the archive. "
            f"Open manually: {source.get('url')}\nPreview: {preview}"
        )
    temporary.replace(destination)
    write_source_manifest(source, destination.parent, "downloaded")


def download_roboflow(source: dict[str, Any], *, force: bool) -> None:
    destination = Path(source["destination"])
    if destination.exists() and any(destination.rglob("data.yaml")) and not force:
        write_source_manifest(source, destination, "present")
        return
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            f"{source['name']} requires ROBOFLOW_API_KEY. "
            "Export it in your shell or add it to .env."
        )
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise SystemExit("install data dependencies first: pip install -e '.[data]'") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(source["workspace"]).project(source["project"])
    dataset = project.version(int(source["version"])).download(
        source.get("format", "yolov8"),
        location=str(destination),
        overwrite=force or destination.exists(),
    )
    root = Path(getattr(dataset, "location", destination))
    write_source_manifest(source, root, "downloaded")


def load_dotenv_if_present() -> None:
    path = Path(".env")
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download external Hurgor dataset/model assets")
    parser.add_argument("--config", type=Path, default=Path("configs/external_datasets.json"))
    parser.add_argument("--only", action="append", help="source name; can be passed more than once")
    parser.add_argument("--skip-roboflow", action="store_true")
    parser.add_argument("--skip-git", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-drive", action="store_true")
    parser.add_argument("--update", action="store_true", help="git pull existing git sources")
    parser.add_argument(
        "--force", action="store_true", help="overwrite downloaded file/export targets"
    )
    parser.add_argument("--allow-large-downloads", action="store_true")
    args = parser.parse_args()
    load_dotenv_if_present()
    names = set(args.only) if args.only else None
    config = load_config(args.config)
    for source in selected_sources(config, names):
        kind = source["kind"]
        if kind == "git":
            if not args.skip_git:
                download_git(source, update=args.update)
        elif kind == "roboflow":
            if not args.skip_roboflow:
                download_roboflow(source, force=args.force)
        elif kind == "huggingface_file":
            if not args.skip_models:
                download_huggingface_file(source, force=args.force)
        elif kind == "google_drive_file":
            if not args.skip_drive:
                download_google_drive_file(
                    source, force=args.force, allow_large=args.allow_large_downloads
                )
        else:
            raise SystemExit(f"unsupported source kind: {kind}")
    print("external asset download step complete", file=sys.stderr)


if __name__ == "__main__":
    main()
