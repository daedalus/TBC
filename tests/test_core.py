"""Tests for tbc.core module."""

import os

import pytest
import pytest_mock

from tbc import core


class TestHumanReadableSize:
    """Tests for human_readable_size function."""

    def test_zero_bytes(self) -> None:
        assert core.human_readable_size(0) == "0B"

    def test_bytes(self) -> None:
        assert core.human_readable_size(512) == "512.00 B"

    def test_kilobytes(self) -> None:
        assert core.human_readable_size(1024) == "1.00 KB"

    def test_megabytes(self) -> None:
        assert core.human_readable_size(1048576) == "1.00 MB"

    def test_gigabytes(self) -> None:
        assert core.human_readable_size(1073741824) == "1.00 GB"

    def test_large_value(self) -> None:
        result = core.human_readable_size(1234567890)
        assert "GB" in result


class TestCalculateMD5:
    """Tests for calculate_md5 function."""

    def test_existing_file(self, sample_file: str) -> None:
        result = core.calculate_md5(sample_file)
        assert result is not None
        assert len(result) == 32

    def test_nonexistent_file(self) -> None:
        result = core.calculate_md5("/nonexistent/file.txt")
        assert result is None


class TestGetFileSize:
    """Tests for get_file_size function."""

    def test_existing_file(self, sample_file: str) -> None:
        result = core.get_file_size(sample_file)
        assert result > 0

    def test_file_content_size(self, temp_dir: str) -> None:
        filepath = os.path.join(temp_dir, "test.txt")
        content = "hello world"
        with open(filepath, "w") as f:
            f.write(content)
        assert core.get_file_size(filepath) == len(content.encode())


class TestGetFileSignature:
    """Tests for get_file_signature function."""

    def test_tflite_signature(self, sample_model_file: str) -> None:
        result = core.get_file_signature(sample_model_file, local=True)
        assert result == "TFlite", f"Expected TFlite, got {result}"

    def test_unknown_signature(self, sample_file: str) -> None:
        result = core.get_file_signature(sample_file, local=True)
        assert result == "Unknown"

    def test_onnx_signature(self, temp_dir: str) -> None:
        filepath = os.path.join(temp_dir, "model.onnx")
        with open(filepath, "wb") as f:
            f.write(b"\x4f\x4e\x4e\x58" + b"testdata")
        result = core.get_file_signature(filepath, local=True)
        assert result == "ONNX", f"Expected ONNX, got {result}"

    def test_gguf_signature(self, temp_dir: str) -> None:
        filepath = os.path.join(temp_dir, "model.gguf")
        with open(filepath, "wb") as f:
            f.write(b"GGUF\x03\x00\x00\x00" + b"testdata")
        result = core.get_file_signature(filepath, local=True)
        assert result == "GGUF", f"Expected GGUF, got {result}"


class TestIsCowFilesystem:
    """Tests for is_cow_filesystem function."""

    def test_non_posix_raises(self) -> None:
        if os.name == "posix":
            pytest.skip("Only applicable to non-POSIX systems")
        with pytest.raises(EnvironmentError):
            core.is_cow_filesystem("/")

    def test_root_path(self) -> None:
        result = core.is_cow_filesystem("/")
        assert isinstance(result, bool)


class TestScanFiles:
    """Tests for scan_files function."""

    def test_scan_model_file(self, sample_model_file: str) -> None:
        core.FOUND.clear()
        core.scan_files([sample_model_file], local=True)
        assert sample_model_file in core.FOUND

    def test_scan_apk_no_extraction(self, sample_apk_file: str) -> None:
        core.FOUND.clear()
        core.scan_files([sample_apk_file], local=True)
        assert len(core.FAILED) > 0

    def test_duplicate_file(self, sample_model_file: str) -> None:
        core.FOUND.clear()
        core.scan_files([sample_model_file], local=True)
        core.scan_files([sample_model_file], local=True)
        assert len(core.FOUND) == 1


class TestExportSummaryToCsv:
    """Tests for export_summary_to_csv function."""

    def test_export_empty(self, temp_dir: str) -> None:
        csv_path = os.path.join(temp_dir, "summary.csv")
        core.export_summary_to_csv([], csv_path)
        assert os.path.exists(csv_path)

    def test_export_with_data(self, temp_dir: str) -> None:
        csv_path = os.path.join(temp_dir, "summary.csv")
        data = [
            {
                "MD5": "abc123",
                "Size": "1.00 MB",
                "File Path": "/test/model.tflite",
                "Dst File Path": "/dst/model.tflite",
            }
        ]
        core.export_summary_to_csv(data, csv_path)
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            content = f.read()
            assert "abc123" in content
            assert "model.tflite" in content


class TestCleanupTmpDirectory:
    """Tests for cleanup_tmp_directory function."""

    def test_cleanup_nonexistent(self, temp_dir: str) -> None:
        original_basedir = core.TMPBASEDIR
        core.TMPBASEDIR = temp_dir
        try:
            core.cleanup_tmp_directory()
        finally:
            core.TMPBASEDIR = original_basedir

    def test_cleanup_existing(self, temp_dir: str) -> None:
        original_basedir = core.TMPBASEDIR
        core.TMPBASEDIR = temp_dir
        extracted_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extracted_dir)
        try:
            core.cleanup_tmp_directory()
            assert not os.path.exists(extracted_dir)
        finally:
            core.TMPBASEDIR = original_basedir


class TestLocalShell:
    """Tests for local_shell function."""

    def test_valid_command(self) -> None:
        result = core.local_shell(["echo", "test"])
        assert result is not False
        assert "test" in result

    def test_invalid_command(self) -> None:
        result = core.local_shell(["nonexistent_command_xyz"])
        assert result is False


class TestAdbShell:
    """Tests for adb_shell function."""

    def test_adb_not_available(self) -> None:
        result = core.adb_shell(["echo", "test"])
        assert result == ""


class TestGetFileSizeError:
    """Tests for get_file_size function error case."""

    def test_nonexistent_file(self, temp_dir: str) -> None:
        filepath = os.path.join(temp_dir, "nonexistent.txt")
        with pytest.raises(FileNotFoundError):
            core.get_file_size(filepath)


class TestScanFilesRemote:
    """Tests for scan_files with remote files."""

    def test_scan_files_remote(self, mocker: pytest_mock.MockerFixture) -> None:
        mocker.patch("tbc.core.get_file_size", return_value=1024)
        mocker.patch("tbc.core.get_file_signature", return_value="TFlite")
        core.FOUND.clear()
        core.scan_files(["/remote/path/model.tflite"], local=False)
        assert "/remote/path/model.tflite" in core.FOUND


class TestScanLocalDirectory:
    """Tests for scan_local_directory function."""

    def test_scan_local_directory(self, temp_dir: str) -> None:
        model_file = os.path.join(temp_dir, "model.tflite")
        with open(model_file, "wb") as f:
            f.write(b"\x14\x00\x00\x00TFL3testdata")
        core.FOUND.clear()
        core.scan_local_directory(temp_dir)
        assert model_file in core.FOUND

    def test_scan_local_directory_apk(self, temp_dir: str) -> None:
        apk_file = os.path.join(temp_dir, "app.apk")
        with open(apk_file, "wb") as f:
            f.write(b"PK\x03\x04" + b"\x00" * 100)
        core.FAILED.clear()
        core.scan_local_directory(temp_dir)
        assert apk_file in core.FAILED


class TestAdbPull:
    """Tests for adb_pull function."""

    def test_adb_pull_not_available(self) -> None:
        result = core.adb_pull("/remote/file", "/local/file")
        assert result is False


class TestGetRemoteMd5:
    """Tests for get_remote_md5 function."""

    def test_get_remote_md5_no_adb(self) -> None:
        result = core.get_remote_md5("/remote/file")
        assert result is None

    def test_get_remote_md5_empty_output(
        self, mocker: pytest_mock.MockerFixture
    ) -> None:
        mocker.patch("tbc.core.adb_shell", return_value="")
        result = core.get_remote_md5("/remote/file")
        assert result is None

    def test_get_remote_md5_with_content(
        self, mocker: pytest_mock.MockerFixture
    ) -> None:
        mocker.patch("tbc.core.adb_shell", return_value="abc123def456  /remote/file")
        result = core.get_remote_md5("/remote/file")
        assert result == "abc123def456"


class TestIsCowFilesystemError:
    """Additional tests for is_cow_filesystem."""

    def test_non_posix(self, mocker: pytest_mock.MockerFixture) -> None:
        mocker.patch("os.name", "nt")
        with pytest.raises(OSError):
            core.is_cow_filesystem("/")


class TestCli:
    """Tests for CLI module."""

    def test_cli_module_imports(self) -> None:
        from tbc import cli

        assert cli.main is not None

    def test_cli_invokes_core_run_with_file(
        self, mocker: pytest_mock.MockerFixture
    ) -> None:
        from click.testing import CliRunner

        from tbc import cli

        mocker.patch("tbc.core.run")
        runner = CliRunner()
        runner.invoke(cli.main, ["--file", "/test/file"])

    def test_cli_invokes_core_run_with_local_dir(
        self, mocker: pytest_mock.MockerFixture
    ) -> None:
        from click.testing import CliRunner

        from tbc import cli

        mocker.patch("tbc.core.run")
        runner = CliRunner()
        runner.invoke(cli.main, ["--local-dir", "/test/dir"])

    def test_cli_invokes_core_run_with_cleanup(
        self, mocker: pytest_mock.MockerFixture
    ) -> None:
        from click.testing import CliRunner

        from tbc import cli

        mocker.patch("tbc.core.run")
        runner = CliRunner()
        runner.invoke(cli.main, ["--cleanup"])

    def test_cli_invokes_core_run_with_export_csv(
        self, mocker: pytest_mock.MockerFixture
    ) -> None:
        from click.testing import CliRunner

        from tbc import cli

        mocker.patch("tbc.core.run")
        runner = CliRunner()
        runner.invoke(cli.main, ["--export-csv", "output.csv"])


class TestGetDeviceInfo:
    """Tests for get_device_info function."""

    def test_get_device_info_no_adb(self, mocker: pytest_mock.MockerFixture) -> None:
        mocker.patch("tbc.core.adb_shell", return_value="")
        result = core.get_device_info()
        assert result["serial"] == "Unknown"
        assert result["model"] == "Unknown"
        assert result["manufacturer"] == "Unknown"


class TestFindMlModels:
    """Tests for find_ml_models function."""

    def test_find_ml_models_no_adb(self, mocker: pytest_mock.MockerFixture) -> None:
        mocker.patch("tbc.core.adb_shell", return_value="")
        mocker.patch("tbc.core.get_device_info")
        mocker.patch("tbc.core.get_file_signature", return_value="Unknown")
        core.FOUND.clear()
        result = core.find_ml_models()
        assert result == set()


class TestRun:
    """Tests for run function."""

    def test_run_with_file(self, mocker: pytest_mock.MockerFixture) -> None:
        mocker.patch("tbc.core.scan_files")
        core.FOUND.clear()
        core.run(file="/test/file")
        core.scan_files.assert_called_once()

    def test_run_with_local_dir(self, mocker: pytest_mock.MockerFixture) -> None:
        mocker.patch("tbc.core.scan_local_directory")
        core.FOUND.clear()
        core.run(local_dir="/test/dir")
        core.scan_local_directory.assert_called_once_with("/test/dir")

    def test_run_no_files_found(self, mocker: pytest_mock.MockerFixture) -> None:
        mocker.patch("tbc.core.get_device_info")
        mocker.patch("tbc.core.find_ml_models", return_value=set())
        core.FOUND.clear()
        result = core.run(export_csv="test.csv", cleanup=True)
        assert result is None

    def test_run_with_cleanup(self, mocker: pytest_mock.MockerFixture) -> None:
        mocker.patch("tbc.core.get_device_info")
        mocker.patch("tbc.core.find_ml_models", return_value=set())
        mocker.patch("tbc.core.cleanup_tmp_directory")
        core.FOUND.clear()
        core.run(cleanup=True)
        core.cleanup_tmp_directory.assert_called_once()
