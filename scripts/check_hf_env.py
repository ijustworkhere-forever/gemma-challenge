#!/usr/bin/env python3
"""
Validate local environment variables needed for Hugging Face contest helpers.
"""

import os
from pathlib import Path
import shutil
import sys


REQUIRED_ENV = ["AGENT_ID"]
OPTIONAL_ENV = ["HF_TOKEN", "API"]
REQUIRED_BINARIES = ["hf"]


def has_cached_hf_token() -> bool:
    token_paths = [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]
    return any(path.exists() for path in token_paths)


def main() -> int:
    missing_env = [name for name in REQUIRED_ENV if not os.getenv(name)]
    missing_bins = [name for name in REQUIRED_BINARIES if shutil.which(name) is None]
    has_token = bool(os.getenv("HF_TOKEN")) or has_cached_hf_token()

    print("Gemma Challenge environment check")
    print("Required environment:")
    for name in REQUIRED_ENV:
        print(f"- {name}: {'set' if os.getenv(name) else 'missing'}")

    print("Authentication:")
    if os.getenv("HF_TOKEN"):
        print("- HF_TOKEN: set")
    elif has_cached_hf_token():
        print("- cached Hugging Face token: found")
    else:
        print("- Hugging Face token: missing")

    print("Optional environment:")
    for name in OPTIONAL_ENV:
        print(f"- {name}: {'set' if os.getenv(name) else 'not set'}")

    print("Required commands:")
    for name in REQUIRED_BINARIES:
        print(f"- {name}: {'found' if shutil.which(name) else 'missing'}")
    has_buckets = shutil.which("hf") is not None and os.system("hf buckets --help >/dev/null 2>&1") == 0
    print(f"- hf buckets: {'found' if has_buckets else 'missing'}")

    if missing_env or missing_bins or not has_token or not has_buckets:
        if missing_env:
            print("\nMissing env vars: " + ", ".join(missing_env))
        if missing_bins:
            print("Missing commands: " + ", ".join(missing_bins))
        if not has_token:
            print("Missing Hugging Face auth. Set HF_TOKEN or run: hf auth login")
        if not has_buckets:
            print("Missing hf buckets. Upgrade with: pip install -U huggingface_hub")
        return 1

    print("\nOK: local helper prerequisites are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
