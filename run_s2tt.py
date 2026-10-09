"""Run the checked-out source with the container's Python, without installing this project."""
from pathlib import Path
import sys


def main():
    if sys.version_info < (3, 10):
        print("Python 3.10+ is required; the reference CUDA runtime uses Python 3.12.", file=sys.stderr)
        return 1
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    try:
        from s2tt.cli import main as application
    except ModuleNotFoundError as error:
        print(f"Missing Python dependency: {error.name}. Add it to your existing python3 container environment.",
              file=sys.stderr)
        return 1
    return application()


if __name__ == "__main__":
    raise SystemExit(main())
