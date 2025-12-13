#!/usr/bin/env python3
"""
Kaggle CLI Wrapper
.envファイルを自動的に読み込んでKaggle CLIを実行します
"""
import os
import sys
import subprocess
from pathlib import Path


def load_env_file():
    """Load environment variables from .env file"""
    # Look for .env in current working directory (project root)
    env_path = Path.cwd() / ".env"

    if not env_path.exists():
        print(f"Warning: .env file not found at {env_path}", file=sys.stderr)
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse KEY=VALUE format
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()


def main():
    """Main entry point for the Kaggle CLI wrapper"""
    # Load environment variables from .env
    load_env_file()

    # Run kaggle CLI with all arguments
    cmd = ['kaggle'] + sys.argv[1:]

    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print("Error: kaggle command not found. Make sure kaggle package is installed.", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
