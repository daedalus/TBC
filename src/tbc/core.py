#!/usr/bin/python3

import csv
import hashlib
import os
import shutil
import subprocess
from collections.abc import Sequence
from typing import Any

from colorama import Fore, Style, init

init(autoreset=True)

ML_MODEL_EXTENSIONS: list[str] = [
    ".lite",
    ".tflite",
    ".onnx",
    ".pt",
    ".pth",
    ".pb",
    ".h5",
    ".hdf5",
    ".caffemodel",
    ".weights",
    ".mlmodel",
    ".gguf",
    ".safetensors",
]
APK_EXTENSIONS: list[str] = [".apk"]

TMPBASEDIR = "./tmp"
MODELSDIR = "./models_found"

FOUND: set[str] = set()
FAILED: set[str] = set()

SEARCH_PATHS: list[str] = [
    "/data/data",
    "/data/local/tmp",
    "/sdcard/Android/data",
    "/sdcard/Download",
    "/sdcard/Documents",
    "/sdcard/",
    "/",
]

ML_MODEL_SIGNATURES: dict[bytes, str] = {
    b"\x14\x00\x00\x00TFL3": "TFlite",
    b" \x00\x00\x00TFL3": "TFlite",
    b"\x1c\x00\x00\x00TFL3": "TFlite",
    b"\x18\x00\x00\x00TFL3": "TFlite",
    b"-\x00\x00\x80\xbfb\xbf\x01": "TFlite",
    b"\x4f\x4e\x4e\x58": "ONNX",
    b"\x80\x02\x63\x6e\x6e": "PyTorch",
    b"\x08\x01\x12": "TensorFlow .pb",
    b'\n"\r\x00\x00 A\r': "TensorFlow .pb",
    b"\x89HDF\r\n\x1a\n": "HDF5 (Keras)",
    b"bplist": "Apple CoreML",
    b"GGUF\x03\x00\x00\x00": "GGUF",
    b'{\n  "metadata": {': "SafeTensors",
}


def is_cow_filesystem(path: str = "/") -> bool:
    """Check if the filesystem at the given path is a COW filesystem."""
    if os.name != "posix":
        raise OSError("This function is designed for Linux-based systems.")

    with open("/proc/mounts") as f:
        mounts = f.readlines()

    for mount in mounts:
        device, mount_point, fs_type, *_options = mount.split()
        if fs_type in ["btrfs", "zfs", "overlay", "xfs"]:
            if path.startswith(mount_point):
                return True
    return False


def local_shell(command: Sequence[str]) -> str | bool:
    """Execute a local shell command."""
    try:
        return subprocess.check_output(
            command, stderr=subprocess.DEVNULL, universal_newlines=True
        )
    except Exception:
        return False


def adb_shell(command: Sequence[str]) -> str:
    """Run an ADB shell command on the connected Android device."""
    try:
        result = subprocess.check_output(
            ["adb", "shell"] + list(command),
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
        return result.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def adb_pull(remote_path: str, local_path: str) -> bool:
    """Pull a file from the Android device to the local machine using ADB."""
    try:
        subprocess.check_output(
            ["adb", "pull", remote_path, local_path], stderr=subprocess.DEVNULL
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def calculate_md5(file_path: str) -> str | None:
    """Calculate the MD5 hash of a file."""
    if not os.path.exists(file_path):
        return None
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_remote_md5(remote_path: str) -> str | None:
    """Calculate the MD5 hash of a file on the Android device."""
    md5_output = adb_shell(["md5sum", f'"{remote_path}"'])
    if not md5_output:
        return None
    return md5_output.split()[0]


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.2f} {size_names[i]}"


def get_file_size(file_path: str) -> int:
    """Get the size of a file."""
    return os.stat(file_path).st_size


def get_file_signature(file_path: str, local: bool = False) -> str:
    """Extract the first 8 bytes and compare against known ML model signatures."""
    signature_bytes: bytes = b""
    if not local:
        hex_str = adb_shell(["xxd", "-l 8", "-p", file_path])
        if hex_str:
            try:
                signature_bytes = bytes.fromhex(hex_str)
            except ValueError:
                pass
    else:
        result = local_shell(["xxd", "-l 8", "-p", file_path])
        if result is not False:
            try:
                signature_bytes = bytes.fromhex(str(result))
            except ValueError:
                pass

    if not signature_bytes:
        try:
            with open(file_path, "rb") as f:
                signature_bytes = f.read(8)
        except OSError:
            return "Unknown"

    for magic_bytes, model_type in ML_MODEL_SIGNATURES.items():
        if signature_bytes.startswith(magic_bytes):
            return model_type
    return "Unknown"


def scan_files(files: Sequence[str], local: bool = False) -> None:
    """Scan a list of files for ML model or APK extensions."""
    for file in files:
        if any(file.lower().endswith(ext) for ext in ML_MODEL_EXTENSIONS):
            if file not in FOUND:
                if not local:
                    size = get_file_size(file)
                    model_type = get_file_signature(file)
                else:
                    size = os.path.getsize(file)
                    model_type = get_file_signature(file, local=True)
                if model_type != "Unknown":
                    print(
                        f"{Fore.GREEN}[+] Found possible ML Model: {file} "
                        f"({human_readable_size(size)}) [Type: {model_type}]"
                        f"{Style.RESET_ALL}"
                    )
                    FOUND.add(file)
        elif any(file.lower().endswith(ext) for ext in APK_EXTENSIONS):
            extract_and_scan_apk(file, local=local)


def extract_and_scan_apk(apk_path: str, local: bool = False) -> None:
    """Pull an APK from the device, extract it, and search for ML models."""
    print(f"{Fore.BLUE}[*] Processing APK: {apk_path}{Style.RESET_ALL}")

    apk_name = os.path.basename(apk_path)
    local_apk_path = os.path.join(TMPBASEDIR, apk_name)

    local_md5: str | None = None
    remote_md5: str | None = None

    if not local:
        remote_md5 = get_remote_md5(apk_path)
        if not remote_md5:
            print(
                f"{Fore.RED}[-] Failed to calculate MD5 for remote APK: "
                f"{apk_path}{Style.RESET_ALL}"
            )
            return

        local_md5 = calculate_md5(local_apk_path)

        if local_md5 and local_md5 == remote_md5:
            print(
                f"{Fore.YELLOW}[*] APK already exists in {TMPBASEDIR} with matching "
                f"MD5. Skipping pull.{Style.RESET_ALL}"
            )
        else:
            print(f"{Fore.BLUE}[*] Pulling APK to .{TMPBASEDIR}...{Style.RESET_ALL}")
            if not adb_pull(apk_path, local_apk_path):
                print(f"{Fore.RED}[-] Failed to pull APK: {apk_path}{Style.RESET_ALL}")
                return
    else:
        local_apk_path = apk_path

    print(
        f"{Fore.BLUE}[*] Extracting APK: remote: {apk_path} local: "
        f"{local_apk_path} {Style.RESET_ALL}"
    )

    extracted_dir = os.path.join(TMPBASEDIR, "extracted", apk_name) + "/"
    os.makedirs(extracted_dir, exist_ok=True)
    try:
        subprocess.check_output(
            ["unzip", "-u", local_apk_path, "-d", extracted_dir],
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
    except subprocess.CalledProcessError:
        print(f"{Fore.RED}[-] Failed to extract APK: {local_apk_path}{Style.RESET_ALL}")
        FAILED.add(apk_path)
        return

    for root, _, files in os.walk(extracted_dir):
        for file in files:
            file_path = os.path.join(root, file)
            scan_files([file_path], local=True)


def find_ml_models() -> set[str]:
    """Search for ML model weight files on an Android device."""
    print(
        f"{Fore.BLUE}[*] Searching for ML model files on the device...\n"
        f"{Style.RESET_ALL}"
    )

    for path in SEARCH_PATHS:
        print(f"{Fore.CYAN}[*] Checking {path}...{Style.RESET_ALL}")
        command = ["find", f'"{path}"', "-type", "f"]
        files = adb_shell(command).split("\n")

        for file in files:
            if any(
                file.lower().endswith(ext)
                for ext in ML_MODEL_EXTENSIONS + APK_EXTENSIONS
            ):
                scan_files([file])
    return FOUND


def scan_local_directory(directory: str) -> None:
    """Scan a local directory for APKs and ML models."""
    print(f"{Fore.BLUE}[*] Scanning local directory: {directory}{Style.RESET_ALL}")
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if any(file.lower().endswith(ext) for ext in APK_EXTENSIONS):
                extract_and_scan_apk(file_path, local=True)
            elif any(file.lower().endswith(ext) for ext in ML_MODEL_EXTENSIONS):
                scan_files([file_path], local=True)


def get_device_info() -> dict[str, str]:
    """Retrieve the Android device's serial number, model, and manufacturer."""
    serial = adb_shell(["getprop", "ro.serialno"])
    model = adb_shell(["getprop", "ro.product.model"])
    manufacturer = adb_shell(["getprop", "ro.product.manufacturer"])

    print(f"\n{Fore.BLUE}[*] Device Information:{Style.RESET_ALL}")
    print(
        f"    {Fore.CYAN}Serial Number: {serial if serial else 'Unknown'}"
        f"{Style.RESET_ALL}"
    )
    print(f"    {Fore.CYAN}Model: {model if model else 'Unknown'}{Style.RESET_ALL}")
    print(
        f"    {Fore.CYAN}Manufacturer: {manufacturer if manufacturer else 'Unknown'}"
        f"{Style.RESET_ALL}\n"
    )

    return {
        "serial": serial if serial else "Unknown",
        "model": model if model else "Unknown",
        "manufacturer": manufacturer if manufacturer else "Unknown",
    }


def export_summary_to_csv(
    summary_data: list[dict[str, Any]], csv_filename: str = "summary.csv"
) -> None:
    """Export the summary of found ML models to a CSV file."""
    fieldnames = ["MD5", "Size", "File Path", "Dst File Path"]

    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)

    print(f"{Fore.BLUE}[*] Summary exported to {csv_filename}{Style.RESET_ALL}")


def local_cp(src: str, dst: str, is_cow: bool = True) -> None:
    """Copy a file with reflink if on COW filesystem."""
    if is_cow:
        local_shell(["cp", "-u", "--reflink", src, dst])
    else:
        local_shell(["cp", "-u", src, dst])


def cleanup_tmp_directory() -> None:
    """Clean up the tmp/extracted directory."""
    path = os.path.join(TMPBASEDIR, "extracted")
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        print(
            f"{Fore.YELLOW}[*] {path} does not exist. No cleanup needed.{Style.RESET_ALL}"
        )


def run(
    file: str | None = None,
    local_dir: str | None = None,
    export_csv: str | None = None,
    cleanup: bool = False,
) -> None:
    """Main entry point for the TBC tool."""
    is_cow = is_cow_filesystem(os.getcwd())

    os.makedirs(TMPBASEDIR, exist_ok=True)

    if file:
        scan_files([file])
    elif local_dir:
        scan_local_directory(local_dir)
    else:
        get_device_info()
        find_ml_models()

    if len(FOUND) > 0:
        os.makedirs(MODELSDIR, exist_ok=True)
        print(f"\n{Fore.BLUE}[*] Summary:{Style.RESET_ALL}")

        md5_to_files_map: dict[str, set[str]] = {}
        summary_data: list[dict[str, Any]] = []

        for ml_file in FOUND:
            file_md5 = calculate_md5(ml_file)
            if file_md5 is None:
                continue
            md5_model_dir = os.path.join(MODELSDIR, file_md5)
            os.makedirs(md5_model_dir, exist_ok=True)

            if file_md5 not in md5_to_files_map:
                md5_to_files_map[file_md5] = set()
            md5_to_files_map[file_md5].add(ml_file)

        for file_md5, files in md5_to_files_map.items():
            files_list = list(files)
            md5_model_dir = os.path.join(MODELSDIR, file_md5)
            ext = os.path.basename(files_list[0]).split(".")[-1]
            hrs = human_readable_size(get_file_size(files_list[0]))
            print(f"{Fore.BLUE} With MD5: {file_md5} and size: {hrs}{Style.RESET_ALL}")
            model = os.path.join(md5_model_dir, "model." + ext)

            local_cp(files_list[0], model, is_cow)

            for ml_file in files_list:
                dst_file_basename = os.path.basename(ml_file)
                dst_file_path = os.path.join(md5_model_dir, dst_file_basename)
                local_cp(ml_file, dst_file_path, is_cow)
                summary_data.append(
                    {
                        "MD5": file_md5,
                        "Size": hrs,
                        "File Path": ml_file,
                        "Dst File Path": dst_file_path,
                    }
                )
                print(
                    f"   {Fore.CYAN} Found possible ML Model: [{ml_file}]. "
                    f"{Style.RESET_ALL}"
                )

        print(f"{Fore.YELLOW}[-] Summary of failed APKs: {Style.RESET_ALL}")
        for apk_file in FAILED:
            print(f"   {Fore.RED} Failed APK: [{apk_file}]. {Style.RESET_ALL}")

        if export_csv is None:
            export_csv = "model_report.csv"
        export_summary_to_csv(summary_data, export_csv)

    if cleanup:
        cleanup_tmp_directory()
