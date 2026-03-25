"""CLI module for The Brain Collector."""

import click


@click.command()
@click.option(
    "--file",
    type=str,
    help="Specify a single file to scan.",
)
@click.option(
    "--local-dir",
    type=str,
    help="Specify a local directory to scan for APKs and ML models.",
)
@click.option(
    "--export-csv",
    type=str,
    default=None,
    help="Specify a filename to export the summary to CSV.",
)
@click.option(
    "--cleanup",
    is_flag=True,
    help="Clean up the tmp/ directory after execution.",
)
def main(
    file: str | None,
    local_dir: str | None,
    export_csv: str | None,
    cleanup: bool,
) -> None:
    """Search for ML models on an Android device via ADB."""
    from tbc import core

    core.run(file=file, local_dir=local_dir, export_csv=export_csv, cleanup=cleanup)
