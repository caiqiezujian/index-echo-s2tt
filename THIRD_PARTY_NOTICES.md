# Third-party provenance

The Index-Echo component loading, connector definitions, feature configuration and generation layout in `src/s2tt/backends/index_echo.py` are adapted from IndexTeam's Apache-2.0 model inference recipes. Source repositories, exact audited commits, and snapshot checksums are recorded in `configs/upstream_manifest.json` and `configs/model_snapshots.json`.

- Index LLM Team / bilibili Index-Translate: https://github.com/bilibili/Index-Translate — Apache-2.0.
- IndexTeam Index-Echo-S2TT model exports: https://huggingface.co/IndexTeam/Index-Echo-S2TT-2B and the 9B sibling — Apache-2.0 as declared by the publisher. Model weights are downloaded separately.
- FBK / hlt-mt simulstream: https://github.com/hlt-mt/simulstream — Apache-2.0. Optional adapter targets its public interface; its source is not bundled.

Model artifacts are not included in this repository. Python dependency licenses remain those of their respective publishers. See `LICENSE` for this project's Apache-2.0 terms.
