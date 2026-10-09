# Repository Guidelines

## Project Structure & Module Organization

This repository implements an Index-Echo English-to-Chinese streaming prototype. Read `docx/01_技术方案.md`, `docx/02_Codex实施任务书.md`, `docx/03_测试与验收规范.md`, and `docx/04_调查证据与学习资料.md` before architectural changes. See `docx/06_AutoDL_5090环境准备.md` for server setup and README for executable interfaces.

`src/s2tt/` contains `audio/`, `core/`, `backends/`, `devices/`, `policies/`, `parsing/`, `transports/`, `evaluation/`, and browser assets in `web/`. `tests/` covers contracts; `deploy/` supplies the project installer. `scripts/` provides preparation utilities, `envs/` separates candidate dependencies, and `configs/` records snapshots. Keep weights, recordings, credentials, generated reports, and unrelated exports outside Git; track model revisions and checksums in manifests.

## Build, Test, and Development Commands

Use an isolated project environment. Distinguish CPU/mock validation from real-model acceptance.

- `python -m pip install -e '.[dev]'`: install the CPU development package.
- `ruff check src tests`: run configured static checks.
- `pytest`: run tests; missing optional hardware produces explicit skips.
- `echo-s2tt serve --backend mock`: demonstrate transport without translating speech.
- `echo-s2tt web`: serve the microphone interface on localhost.
- `echo-s2tt serve --config configs/5090_2b.json`: load model paths and runtime settings from local JSON; use the same configuration for `web`.
- `python scripts/download_echo.py --size 2B --root /path/to/models --report-dir reports`: download and verify a fixed ModelScope snapshot.

Maintain separate CUDA, Ascend, and evaluation dependencies. Record actual installed versions; unverified combinations remain candidates.

Inference uses verified local artifacts and offline framework flags. Missing files must fail explicitly; downloading is a separate preparation operation.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, and `PascalCase` classes. Use direction-neutral `source_text` and `target_text`. Keep SessionCore hardware-independent; isolate platform imports in device/backend modules. Ruff provides static checks.

## Testing Guidelines

Follow the acceptance specification: unit/property, contract, real-model, NVIDIA, Ascend, and end-to-end tests. Use pytest and `test_<behavior>.py` names. No numeric coverage target is established.

Verify causal audio visibility, explicit corrections, stale-result rejection, and complete End handling. Label mocks with `model_kind=mock`. Missing hardware or weights must produce BLOCKED/SKIPPED with reasons, never model PASS. Preserve raw logs and failure examples.

## Commit & Pull Request Guidelines

Use concise imperative subjects, optionally prefixed by task ID, e.g., `T04: Add causal audio ledger`.

PRs should describe scope, relevant task/issue, executed commands, PASS/FAIL/BLOCKED results, environment/model revisions, and evidence paths. Include screenshots for UI changes.

## Engineering Constraints

Preserve user changes. Never expose future audio to online inference, silently rewrite committed text, or substitute another model/API while claiming Echo success. Obtain authorization for system-environment changes, data export, paid operations, or destructive actions.
