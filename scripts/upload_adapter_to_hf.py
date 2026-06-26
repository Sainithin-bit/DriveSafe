#!/usr/bin/env python3
"""Upload the DriveSafe LLaMA-Adapter checkpoint to Hugging Face."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, upload_file

from drivesafe.constants import HF_ADAPTER_REPO, HF_ADAPTER_WEIGHT_FILE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload DriveSafe adapter weights to Hugging Face")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Local path to DriveSafe_LLaMA_Adapter_8B.pth",
    )
    parser.add_argument("--repo-id", default=HF_ADAPTER_REPO)
    parser.add_argument(
        "--remote-filename",
        default=HF_ADAPTER_WEIGHT_FILE,
        help="Filename on the Hugging Face Hub",
    )
    return parser.parse_args()


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token

    proc = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=huggingface.co\n\n",
        text=True,
        capture_output=True,
        check=False,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise SystemExit("Hugging Face token not found. Run `huggingface-cli login` or set HF_TOKEN.")


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    token = get_hf_token()
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True, private=False)

    upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo=args.remote_filename,
        repo_id=args.repo_id,
        repo_type="model",
        token=token,
        commit_message=f"Add {args.remote_filename}",
    )
    print(f"Uploaded {checkpoint} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
