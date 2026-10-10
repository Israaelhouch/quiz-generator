"""CI installs `requirements-dev.txt`, not the lockfile.

A developer's virtualenv is built from `requirements.lock.txt` and therefore
carries packages the gate never installs. When a module starts importing one
of those, every local check passes and CI fails on an ImportError — which is
exactly what happened when `pydantic_settings` arrived with the typed Settings
object: it was added to `requirements.txt`, not to the file the gate reads.

This test reads the imports out of the source tree and fails if the gate could
not satisfy one.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Import name -> distribution name, where the two differ.
_DISTRIBUTION_OF = {
    "yaml": "pyyaml",
    "pydantic_settings": "pydantic-settings",
    "dotenv": "python-dotenv",
    "google": "google-genai",
    "sentence_transformers": "sentence-transformers",
    "sklearn": "scikit-learn",
}

# Imported by modules the gate never executes: the heavy adapters are mocked in
# tests, so CI does not install torch, chromadb or the provider SDKs. Keeping
# them out is deliberate — see the header of .github/workflows/ci.yml.
_NOT_NEEDED_BY_THE_GATE = {
    # Serving only: `uvicorn` is imported inside api/__main__.main(), and
    # `dotenv` inside a try/except in the judge script, which says so. Neither
    # is reached by the suite, and the clean-environment run proves it.
    "uvicorn",
    "dotenv",
    "chromadb",
    "sentence_transformers",
    "torch",
    "google",
    "groq",
    "ollama",
    "openai",
    "tqdm",
}


def _third_party_imports(package_dir: Path) -> set[str]:
    """Top-level modules imported by `package_dir`, excluding stdlib and self."""
    found: set[str] = set()
    for path in package_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])

    local = {"quiz_generator", "scripts", "tests"}
    return {name for name in found if name not in sys.stdlib_module_names and name not in local}


def _declared_in(requirements: Path) -> set[str]:
    """Distribution names pinned in a requirements file."""
    declared = set()
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name = line.split("==")[0].split(">=")[0].split("[")[0].strip()
        declared.add(name.lower().replace("_", "-"))
    return declared


def test_every_import_the_gate_runs_is_installed_by_the_gate() -> None:
    """Fails the moment a module imports something CI would not have."""
    imports = _third_party_imports(ROOT / "src" / "quiz_generator")
    imports |= _third_party_imports(ROOT / "scripts")
    needed = {
        _DISTRIBUTION_OF.get(name, name).lower().replace("_", "-")
        for name in imports - _NOT_NEEDED_BY_THE_GATE
    }

    declared = _declared_in(ROOT / "requirements-dev.txt")
    missing = sorted(needed - declared)

    assert not missing, (
        f"imported by src/ or scripts/ but not in requirements-dev.txt: {missing}. "
        "CI installs that file, so the gate will fail on an ImportError even though "
        "a local run passes."
    )


def test_the_runtime_requirements_declare_what_the_service_imports() -> None:
    """`requirements.txt` is the statement of intent; it must not fall behind
    either, or a container build resolves without something the code needs."""
    imports = _third_party_imports(ROOT / "src" / "quiz_generator")
    needed = {_DISTRIBUTION_OF.get(n, n).lower().replace("_", "-") for n in imports}

    declared = _declared_in(ROOT / "requirements.txt")
    missing = sorted(needed - declared)

    assert not missing, f"imported by src/ but not in requirements.txt: {missing}"
