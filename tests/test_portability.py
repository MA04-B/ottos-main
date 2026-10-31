import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_shared_conda_definition():
    environment = yaml.safe_load((ROOT / "environment.yml").read_text())
    assert environment["name"] == "otoslotto-ai"
    assert "python=3.11" in environment["dependencies"]
    assert not (ROOT / "requirements_windows.txt").exists()
    assert not (ROOT / "requirements_linux.txt").exists()


def test_linux_entry_points_parse_and_resolve_own_directory():
    files = ["conda_linux.sh", "install_linux.sh", "start_linux.sh", "update_linux.sh"]
    subprocess.run(["bash", "-n", *(str(ROOT / f) for f in files)], check=True)
    for name in files:
        assert (ROOT / name).stat().st_mode & 0o111
        assert "BASH_SOURCE[0]" in (ROOT / name).read_text()


def test_windows_static_contract():
    helper = (ROOT / "conda_windows.bat").read_text()
    assert 'cd /d "%~dp0"' in helper
    assert 'call "%CONDA_CMD%" activate otoslotto-ai' in helper
    assert "%~dp0environment.yml" in helper
    assert "miniconda3" in helper and "anaconda3" in helper
    assert "env list |" not in helper
    for name in ("install_windows.bat", "start_windows.bat", "update_windows.bat"):
        script = (ROOT / name).read_text()
        assert 'call "%~dp0conda_windows.bat"' in script
        assert "if errorlevel 1 goto fail" in script
        assert "pause" in script
    start = (ROOT / "start_windows.bat").read_text()
    assert "--server.address 127.0.0.1 --server.port 8501" in start
