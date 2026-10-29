"""Command-line entry point for preparing and releasing a dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from publish.prepare import prepare
from publish.release import release


def main(argv: list[str] | None = None) -> None:
    """Parse ``prepare`` / ``release`` and run the requested command."""
    parser = argparse.ArgumentParser(
        description="Prepare or release a finished PM2.5 dataset.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Upload map assets, create a Zenodo draft, and open the website PR.",
    )
    _add_shared_args(prepare_parser)
    prepare_parser.add_argument(
        "--netcdf",
        type=Path,
        default=None,
        help="Local NetCDF instead of downloading from FINAL_RESULT_BUCKET_NAME.",
    )
    prepare_parser.add_argument(
        "--cv-summary",
        type=Path,
        default=None,
        dest="cv_summary",
        help="Local CV summary CSV instead of downloading from FINAL_RESULT_BUCKET_NAME.",
    )

    release_parser = subparsers.add_parser(
        "release",
        help="Publish Zenodo, switch the map index, and merge the website PR.",
    )
    _add_shared_args(release_parser)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest = prepare(
            args.model_run_ref,
            netcdf_path=args.netcdf,
            cv_summary_path=args.cv_summary,
            manifest_path=args.manifest,
        )
        print(f"Prepared {manifest.model_run_ref}")  # noqa: T201
        print(f"Zenodo draft: {manifest.zenodo_draft_url}")  # noqa: T201
        print(f"Website PR: {manifest.pr_url}")  # noqa: T201
        return
    release(args.model_run_ref, manifest_path=args.manifest)
    print(f"Released {args.model_run_ref}")  # noqa: T201


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model_run_ref", help="Production model-run reference to publish.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to publish-<ref>.json. Defaults to the current directory.",
    )


if __name__ == "__main__":
    main()
