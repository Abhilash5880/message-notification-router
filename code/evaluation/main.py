"""Development-only sample regression evaluation.

This script is intentionally separate from production routing. It reads samples
only to score the current policy; Router/Dataset never load them as evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from router import Router  # noqa: E402


def read_samples(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the local router against solved samples.")
    parser.add_argument("--dataset-dir", type=Path, default=CODE_DIR.parents[0] / "dataset")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report.")
    args = parser.parse_args()
    router = Router(args.dataset_dir)
    samples = read_samples(args.dataset_dir / "sample_messages.csv")
    failures = []
    action_correct = type_correct = 0
    for sample in samples:
        prediction = router.predict(sample)
        action_ok = prediction["action"] == sample["action"]
        type_ok = prediction["message_type"] == sample["message_type"]
        action_correct += action_ok
        type_correct += type_ok
        if not (action_ok and type_ok):
            failures.append({
                "message_id": sample["message_id"],
                "expected": {"action": sample["action"], "message_type": sample["message_type"]},
                "predicted": {key: prediction[key] for key in ("action", "message_type", "reason", "confidence", "evidence_message_ids")},
                "generalizable_diagnosis": diagnose(sample, prediction),
            })
    report = {
        "samples": len(samples),
        "action_accuracy": round(action_correct / len(samples), 4) if samples else 0.0,
        "message_type_accuracy": round(type_correct / len(samples), 4) if samples else 0.0,
        "joint_accuracy": round((len(samples) - len(failures)) / len(samples), 4) if samples else 0.0,
        "failures": failures,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Action accuracy: {action_correct}/{len(samples)} ({report['action_accuracy']:.1%})")
        print(f"Message-type accuracy: {type_correct}/{len(samples)} ({report['message_type_accuracy']:.1%})")
        print(f"Joint accuracy: {len(samples) - len(failures)}/{len(samples)} ({report['joint_accuracy']:.1%})")
        for failure in failures:
            print("- {message_id}: expected {expected[action]}/{expected[message_type]}, "
                  "got {predicted[action]}/{predicted[message_type]}. {generalizable_diagnosis}".format(**failure))
    return 0


def diagnose(sample, prediction) -> str:
    """Failure categories guide reusable policy improvements, never ID-specific fixes."""
    text = (sample.get("message_text") or "").lower()
    if sample.get("media_type") and not text:
        return "Media-only content needs better ASR/vision extraction or a safer fallback prior."
    if sample.get("action") == "mute":
        return "Strengthen behavior/risk weighting for repeated dismissals, opt-outs, or safety signals."
    if sample.get("action") == "notify":
        return "Strengthen directness, deadline, trusted-source, or operational-urgency features."
    if sample.get("message_type") != prediction.get("message_type"):
        return "Refine general semantic category detection before changing routing thresholds."
    return "Rebalance generic relevance and interruption thresholds without sample-specific rules."


if __name__ == "__main__":
    sys.exit(main())
