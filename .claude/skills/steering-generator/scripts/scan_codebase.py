#!/usr/bin/env python3
"""
Codebase Scanner for Steering Generator

Scans the project root and produces a structured Markdown summary
used by Claude to generate CLAUDE.md steering files.

Usage:
    python scan_codebase.py [project_root]

    project_root defaults to the current working directory.
"""

import json
import os
import sys
from pathlib import Path


# Files/dirs to always skip
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    ".env", "dist", "build", ".next", ".nuxt", "coverage", ".mypy_cache",
    ".pytest_cache", ".tox", "htmlcov", "eggs", ".eggs", "*.egg-info",
}

SKIP_FILES = {
    ".DS_Store", "Thumbs.db", "*.pyc", "*.pyo", "*.so", "*.dll",
}

# Configuration / dependency files to read (ordered by priority)
CONFIG_FILES = [
    "README.md", "README.rst", "README.txt",
    "package.json", "pyproject.toml", "setup.py", "setup.cfg",
    "requirements.txt", "requirements-dev.txt", "Pipfile",
    "Gemfile", "go.mod", "Cargo.toml", "pom.xml", "build.gradle",
    "docker-compose.yml", "docker-compose.yaml", "Dockerfile",
    "cdk.json", "serverless.yml", "terraform.tfvars",
    ".env.example", ".env.sample",
    "Makefile",
]

# Source entry points to look for
ENTRY_POINTS = [
    "main.py", "app.py", "index.py", "server.py", "cli.py",
    "manage.py", "wsgi.py", "asgi.py",
    "src/main.py", "src/app.py",
    "index.ts", "index.js", "src/index.ts", "src/index.js",
    "bin/*.ts", "bin/*.js",
]

MAX_FILE_LINES = 60  # Max lines to read from each config file


def should_skip(path: Path) -> bool:
    if path.name in SKIP_DIRS or path.name in SKIP_FILES:
        return True
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    return False


def read_file_head(path: Path, max_lines: int = MAX_FILE_LINES) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        truncated = lines[:max_lines]
        result = "\n".join(truncated)
        if len(lines) > max_lines:
            result += f"\n... (truncated, {len(lines)} lines total)"
        return result
    except Exception as e:
        return f"(Could not read: {e})"


def get_directory_tree(root: Path, max_depth: int = 3) -> str:
    lines = []

    def walk(path: Path, depth: int, prefix: str = ""):
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return

        entries = [e for e in entries if not should_skip(e)]

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{connector}{entry.name}{suffix}")

            if entry.is_dir() and depth < max_depth:
                extension = "    " if is_last else "│   "
                walk(entry, depth + 1, prefix + extension)

    lines.append(f"{root.name}/")
    walk(root, 1)
    return "\n".join(lines)


def detect_languages(root: Path) -> dict:
    ext_counts = {}
    for path in root.rglob("*"):
        if path.is_file() and not should_skip(path):
            ext = path.suffix.lower()
            if ext:
                ext_counts[ext] = ext_counts.get(ext, 0) + 1

    # Map extensions to language names
    lang_map = {
        ".py": "Python", ".ts": "TypeScript", ".js": "JavaScript",
        ".tsx": "TypeScript (React)", ".jsx": "JavaScript (React)",
        ".java": "Java", ".kt": "Kotlin", ".go": "Go",
        ".rs": "Rust", ".rb": "Ruby", ".php": "PHP",
        ".cs": "C#", ".cpp": "C++", ".c": "C",
        ".swift": "Swift", ".scala": "Scala",
        ".tf": "Terraform", ".yaml": "YAML", ".yml": "YAML",
    }

    result = {}
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
        if ext in lang_map:
            lang = lang_map[ext]
            result[lang] = result.get(lang, 0) + count

    return dict(sorted(result.items(), key=lambda x: -x[1])[:8])


def find_test_dirs(root: Path) -> list:
    test_patterns = ["tests", "test", "spec", "__tests__", "e2e"]
    found = []
    for pattern in test_patterns:
        path = root / pattern
        if path.is_dir():
            found.append(str(path.relative_to(root)))
    return found


def find_existing_claude_md(root: Path) -> str | None:
    claude_md = root / "CLAUDE.md"
    if claude_md.exists():
        return claude_md.read_text(encoding="utf-8", errors="replace")
    return None


def extract_package_info(root: Path) -> dict:
    info = {}

    # package.json
    pkg = root / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text())
            info["name"] = data.get("name")
            info["description"] = data.get("description")
            info["scripts"] = data.get("scripts", {})
            deps = list(data.get("dependencies", {}).keys())[:10]
            dev_deps = list(data.get("devDependencies", {}).keys())[:5]
            info["npm_deps"] = deps
            info["npm_dev_deps"] = dev_deps
        except Exception:
            pass

    # pyproject.toml
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text()
        for line in content.splitlines():
            if line.startswith("name ="):
                info["name"] = line.split("=")[1].strip().strip('"')
            if line.startswith("description ="):
                info["description"] = line.split("=")[1].strip().strip('"')

    return info


def scan(root_path: str = ".") -> str:
    root = Path(root_path).resolve()
    output = []

    output.append("# Codebase Scan Report\n")
    output.append(f"**Project root:** `{root}`\n")

    # 1. Package info
    pkg_info = extract_package_info(root)
    if pkg_info.get("name"):
        output.append(f"**Package name:** `{pkg_info['name']}`")
    if pkg_info.get("description"):
        output.append(f"**Description:** {pkg_info['description']}")
    output.append("")

    # 2. Languages
    langs = detect_languages(root)
    if langs:
        output.append("## Languages Detected\n")
        for lang, count in langs.items():
            output.append(f"- {lang}: {count} files")
        output.append("")

    # 3. Directory tree
    output.append("## Directory Structure\n")
    output.append("```")
    output.append(get_directory_tree(root, max_depth=3))
    output.append("```\n")

    # 4. Test directories
    test_dirs = find_test_dirs(root)
    if test_dirs:
        output.append(f"**Test directories:** {', '.join(f'`{d}`' for d in test_dirs)}\n")

    # 5. Config / dependency files
    output.append("## Key Files\n")
    for filename in CONFIG_FILES:
        path = root / filename
        if path.exists() and path.is_file():
            output.append(f"### `{filename}`\n")
            output.append("```")
            output.append(read_file_head(path))
            output.append("```\n")

    # 6. Entry points
    output.append("## Entry Points Found\n")
    found_any = False
    for pattern in ENTRY_POINTS:
        if "*" in pattern:
            matches = list(root.glob(pattern))
        else:
            matches = [root / pattern] if (root / pattern).exists() else []

        for match in matches:
            if match.exists():
                rel = match.relative_to(root)
                output.append(f"- `{rel}`")
                found_any = True
    if not found_any:
        output.append("- None detected automatically")
    output.append("")

    # 7. Existing CLAUDE.md
    existing = find_existing_claude_md(root)
    if existing:
        output.append("## Existing CLAUDE.md\n")
        output.append("```markdown")
        output.append(existing[:3000])
        if len(existing) > 3000:
            output.append("... (truncated)")
        output.append("```\n")
    else:
        output.append("## Existing CLAUDE.md\n")
        output.append("None found — this is a CREATE operation.\n")

    return "\n".join(output)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print(scan(root))


if __name__ == "__main__":
    main()
