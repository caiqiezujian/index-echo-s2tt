from pathlib import Path


def score_text_files(hypotheses, references):
    """Offline paired text quality only. Never used as backend prompt/context."""
    from sacrebleu.metrics import CHRF

    predictions = Path(hypotheses).read_text(encoding="utf-8").splitlines()
    targets = Path(references).read_text(encoding="utf-8").splitlines()
    if not predictions or len(predictions) != len(targets):
        raise ValueError("Require equal nonzero counts of paired text lines")
    metric = CHRF()
    result = metric.corpus_score(predictions, [targets])
    return {"metric": "chrF", "score": result.score, "scale": "0–100",
            "signature": str(metric.get_signature()), "paired_lines": len(predictions),
            "semantic_latency_available": False, "input_mode": "offline_scoring"}
