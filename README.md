# The Brain Collector

> A tool to scan Android devices for ML model weight files via ADB.

[![PyPI](https://img.shields.io/pypi/v/thebraincollector.svg)](https://pypi.org/project/thebraincollector/)
[![Python](https://img.shields.io/pypi/pyversions/thebraincollector.svg)](https://pypi.org/project/thebraincollector/)
[![Coverage](https://codecov.io/gh/dclavijo/TBC/branch/main/graph/badge.svg)](https://codecov.io/gh/dclavijo/TBC)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Install

```bash
pip install thebraincollector
```

## Usage

```python
from tbc import find_ml_models, get_device_info

# Get device information
info = get_device_info()
print(f"Device: {info['model']}")

# Find ML models on connected device
models = find_ml_models()
```

## CLI

```bash
tbc --help
tbc --file /path/to/file
tbc --local-dir /path/to/dir
tbc --export-csv results.csv --cleanup
```

## API

### `get_device_info() -> dict`
Returns device serial number, model, and manufacturer.

### `find_ml_models() -> set`
Searches connected Android device for ML model files.

### `scan_files(files: list[str], local: bool = False) -> None`
Scans files for ML model or APK extensions.

### `human_readable_size(size_bytes: int) -> str`
Converts bytes to human-readable format.

### `calculate_md5(file_path: str) -> str | None`
Calculates MD5 hash of a file.

## Development

```bash
git clone https://github.com/dclavijo/TBC.git
cd TBC
pip install -e ".[test]"

# run tests
pytest

# format
ruff format src/ tests/

# lint
ruff check src/ tests/

# type check
mypy src/
```

## License

MIT License - see LICENSE file for details.
