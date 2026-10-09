#!/usr/bin/env bash
# Run in the cloned repository. Changes only the selected project environment.
set -euo pipefail
INDEX_ROOT="${INDEX_ROOT:-/root/autodl-tmp/index-echo}"
mkdir -p "$INDEX_ROOT"/{models,reports,tmp,cache/pip,cache/modelscope}
export PIP_CACHE_DIR="$INDEX_ROOT/cache/pip"
export MODELSCOPE_CACHE="$INDEX_ROOT/cache/modelscope"
export TMPDIR="$INDEX_ROOT/tmp"
python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Activate an isolated Python 3.12 environment first"'
if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
    echo 'Activate a project venv or conda environment first.' >&2
    exit 1
fi
command -v ffmpeg >/dev/null || { echo 'Install system ffmpeg before continuing.' >&2; exit 1; }
python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r envs/cuda5090/requirements.in
python -m pip install -e '.[dev]'
python -m pip check
python scripts/check_cuda_environment.py --out "$INDEX_ROOT/reports/cuda_environment.json"
python -m pip freeze > "$INDEX_ROOT/reports/cuda5090.freeze.txt"
echo 'Environment checked. Download the 2B snapshot with scripts/download_echo.py.'
