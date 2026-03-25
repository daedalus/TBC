# SPEC.md — The Brain Collector

## Purpose
The Brain Collector (TBC) is a CLI tool designed to search for machine learning model weight files on an Android device via ADB. It scans common directories for known ML model file extensions and signatures and can also extract APK files to analyze their contents for embedded ML models.

## Scope

### In Scope
- Scan Android devices connected via ADB for ML model files
- Support various model file formats: .tflite, .onnx, .pt, .pth, .pb, .h5, .hdf5, .caffemodel, .weights, .mlmodel, .gguf, .safetensors
- Detect files based on extensions and magic byte signatures
- Extract APK files and search for embedded ML models
- Provide device information (serial number, model, manufacturer)
- Display file sizes in human-readable format
- Support local directory scanning
- Export summary to CSV
- Cleanup temporary files

### Not In Scope
- iOS device scanning
- Remote device scanning (non-ADB)
- Model analysis or decompilation
- Auto-download of models
- GUI interface

## Public API / Interface

### CLI Commands

```
tbc [--file FILE] [--local-dir LOCAL_DIR] [--export-csv EXPORT_CSV] [--cleanup]
```

| Option | Type | Description |
|--------|------|-------------|
| `--file` | str | Specify a single file to scan |
| `--local-dir` | str | Specify a local directory to scan |
| `--export-csv` | str | Filename to export summary to CSV (default: model_report.csv) |
| `--cleanup` | flag | Clean up tmp/ directory after execution |

### Public Functions

#### `get_device_info() -> dict`
Returns device serial number, model, and manufacturer.

#### `find_ml_models() -> set`
Searches connected Android device for ML model files. Returns set of found files.

#### `scan_files(files: list[str], local: bool = False) -> None`
Scans a list of files for ML model or APK extensions.

#### `extract_and_scan_apk(apk_path: str, local: bool = False) -> None`
Extracts APK and scans contents for ML models.

#### `scan_local_directory(directory: str) -> None`
Scans a local directory for APKs and ML models.

#### `human_readable_size(size_bytes: int) -> str`
Converts bytes to human-readable format (e.g., "1.23 MB").

#### `calculate_md5(file_path: str) -> str | None`
Calculates MD5 hash of a file.

#### `get_file_signature(file_path: str, local: bool = False) -> str`
Returns file type based on magic bytes.

#### `export_summary_to_csv(summary_data: list[dict], csv_filename: str) -> None`
Exports found models summary to CSV file.

## Data Formats

### Input
- Files from Android device via ADB
- Local files from filesystem

### Output
- Console output with colored findings
- CSV file with columns: MD5, Size, File Path, Dst File Path

### Supported Model Signatures
- TFlite (multiple versions)
- ONNX
- PyTorch
- TensorFlow .pb
- HDF5 (Keras)
- Apple CoreML
- GGUF
- SafeTensors

## Edge Cases

1. **No ADB device connected** - Should handle gracefully with informative message
2. **Permission denied on device** - Skip file, continue scanning
3. **APK extraction fails** - Log to FAILED set, continue
4. **Duplicate files** - Use FOUND set to avoid duplicates
5. **Large files** - Stream MD5 calculation in chunks
6. **Empty device** - No results, no crash
7. **Non-existent local directory** - Handle with error message
8. **File not found during MD5** - Return None
9. **MD5 mismatch between local/remote APK** - Re-pull the file
10. **Filesystem type detection** - Handle non-Linux systems gracefully

## Performance & Constraints

- ADB commands are I/O bound - accept as is
- MD5 calculation uses 4KB chunks for memory efficiency
- File discovery uses `find` command for efficiency
- Target: Python 3.11+
- No heavy third-party dependencies (only colorama, click)
