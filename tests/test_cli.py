"""
Tests for turboagent.cli: init, benchmark, info commands.
"""

import subprocess
import sys
from pathlib import Path

import pytest


class TestCLIInfo:
    def test_info_runs_without_error(self):
        result = subprocess.run(
            [sys.executable, "-m", "turboagent.cli", "info"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Hardware" in result.stdout

    def test_version_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "turboagent.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        from turboagent.version import __version__
        assert __version__ in result.stdout

    def test_no_args_prints_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "turboagent.cli"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "turboagent" in result.stdout.lower() or "usage" in result.stdout.lower()


class TestCLIInit:
    def test_init_creates_project(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "turboagent.cli", "init", "test_project"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0

        project_dir = tmp_path / "test_project"
        assert project_dir.is_dir()
        assert (project_dir / "main.py").is_file()
        assert (project_dir / "requirements.txt").is_file()
        assert (project_dir / "Dockerfile").is_file()
        assert (project_dir / ".gitignore").is_file()
        assert (project_dir / "agents").is_dir()

    def test_init_existing_dir_fails(self, tmp_path):
        (tmp_path / "existing").mkdir()
        result = subprocess.run(
            [sys.executable, "-m", "turboagent.cli", "init", "existing"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(tmp_path),
        )
        assert result.returncode != 0


class TestCLIBenchmark:
    def test_benchmark_runs(self):
        result = subprocess.run(
            [sys.executable, "-m", "turboagent.cli", "benchmark", "--model-size", "7"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert "Benchmark" in result.stdout
