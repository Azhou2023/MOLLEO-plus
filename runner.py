import sys
import subprocess
from pathlib import Path

def run_file(path):
    path = Path(path)

    if not path.is_file():
        print(f"Commands file not found: {path}")
        sys.exit(1)

    with path.open() as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]

    commands = []
    for lineno, line in enumerate(lines, 1):
        if "::" not in line:
            print(f"Line {lineno} missing '::' separator:\n  {line}")
            sys.exit(1)

        dir_part, cmd_part = map(str.strip, line.split("::", 1))
        workdir = Path(dir_part)

        if not workdir.is_dir():
            print(f"Line {lineno} invalid directory: {workdir}")
            sys.exit(1)

        commands.append((workdir, cmd_part))

    for i, (workdir, cmd) in enumerate(commands, 1):
        print(f"\n[{i}/{len(commands)}] ({workdir}) Running: {cmd}")

        result = subprocess.run(
            cmd,
            shell=True,
            cwd=workdir
        )

        if result.returncode != 0:
            print(f"Failed with code {result.returncode}")
            sys.exit(result.returncode)

    print("\nAll commands finished successfully.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_seq.py commands.txt")
        sys.exit(1)

    run_file(sys.argv[1])