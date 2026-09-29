"""Package installation metadata and console entry point (no network)."""

import os
import subprocess
import sys
from importlib.metadata import entry_points, version
from pathlib import Path

import pytest

import sciforge


def test_installed_version_matches_package():
    assert version("sciforge") == sciforge.__version__ == "0.4.0"


def test_console_script_entry_point():
    eps = [ep for ep in entry_points(group="console_scripts") if ep.name == "sciforge"]
    assert len(eps) == 1
    assert eps[0].value == "sciforge.cli:main"
    assert eps[0].load() is sciforge.cli.main


def _env():
    env = dict(os.environ)
    for name in ("NCBI_API_KEY", "SCIFORGE_CONTACT_EMAIL"):
        env.pop(name, None)
    return env


def test_python_dash_m_version_subprocess():
    out = subprocess.run([sys.executable, "-m", "sciforge", "--version"], capture_output=True, text=True,
                         timeout=60, env=_env())
    assert out.returncode == 0 and out.stdout.strip() == "sciforge 0.4.0"


def test_console_script_subprocess():
    script = Path(sys.executable).with_name("sciforge")
    if not script.exists():
        pytest.skip("console script not found next to the interpreter")
    ver = subprocess.run([str(script), "--version"], capture_output=True, text=True, timeout=60, env=_env())
    assert ver.returncode == 0 and ver.stdout.strip() == "sciforge 0.4.0"
    helped = subprocess.run([str(script), "investigate", "--help"], capture_output=True, text=True, timeout=60,
                            env=_env())
    assert helped.returncode == 0 and "--from-year" in helped.stdout
