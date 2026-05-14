#!/usr/bin/env python3
"""
Version management script for tei-annotator.

Updates version numbers across all project files:
- package.json
- pyproject.toml
- tei_annotator/__init__.py

Usage:
    python scripts/version.py [patch|minor|major|VERSION]

Examples:
    python scripts/version.py patch    # 0.1.0 -> 0.1.1
    python scripts/version.py minor    # 0.1.0 -> 0.2.0
    python scripts/version.py major    # 0.1.0 -> 1.0.0
    python scripts/version.py 1.5.2    # Set specific version
"""

import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def parse_version(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:-.*)?$", version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def format_version(major: int, minor: int, patch: int) -> str:
    return f"{major}.{minor}.{patch}"


def increment_version(current: str, bump_type: str) -> str:
    major, minor, patch = parse_version(current)
    if bump_type == "major":
        return format_version(major + 1, 0, 0)
    elif bump_type == "minor":
        return format_version(major, minor + 1, 0)
    elif bump_type == "patch":
        return format_version(major, minor, patch + 1)
    else:
        parse_version(bump_type)  # validate format
        return bump_type


def get_current_version() -> str:
    content = (PROJECT_ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"', content, re.MULTILINE)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")
    return match.group(1)


def update_package_json(new_version: str) -> None:
    package_json = PROJECT_ROOT / "package.json"
    data = json.loads(package_json.read_text())
    data["version"] = new_version
    package_json.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[UPDATED] package.json -> {new_version}")


def update_pyproject_toml(new_version: str) -> None:
    pyproject = PROJECT_ROOT / "pyproject.toml"
    content = pyproject.read_text()
    updated = re.sub(
        r'^version = "[^"]+"',
        f'version = "{new_version}"',
        content,
        flags=re.MULTILINE,
    )
    pyproject.write_text(updated)
    print(f"[UPDATED] pyproject.toml -> {new_version}")


def update_init_py(new_version: str) -> None:
    init_py = PROJECT_ROOT / "tei_annotator" / "__init__.py"
    content = init_py.read_text()
    updated = re.sub(
        r'^__version__ = "[^"]+"',
        f'__version__ = "{new_version}"',
        content,
        flags=re.MULTILINE,
    )
    init_py.write_text(updated)
    print(f"[UPDATED] tei_annotator/__init__.py -> {new_version}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/version.py [patch|minor|major|VERSION]")
        print()
        print("Examples:")
        print("  python scripts/version.py patch    # Increment patch version")
        print("  python scripts/version.py minor    # Increment minor version")
        print("  python scripts/version.py major    # Increment major version")
        print("  python scripts/version.py 1.5.2    # Set specific version")
        sys.exit(1)

    bump_type = sys.argv[1]

    try:
        current_version = get_current_version()
        new_version = increment_version(current_version, bump_type)

        print(f"\nUpdating version: {current_version} -> {new_version}")
        print("=" * 50)

        update_package_json(new_version)
        update_pyproject_toml(new_version)
        update_init_py(new_version)

        subprocess.run(["uv", "lock"], check=True, cwd=PROJECT_ROOT)
        print("[UPDATED] uv.lock")

        subprocess.run(["npm", "install", "--package-lock-only"], check=True, cwd=PROJECT_ROOT)
        print("[UPDATED] package-lock.json")

        print("=" * 50)
        print(f"\n[SUCCESS] All files updated to version {new_version}")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
