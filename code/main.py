"""Terminal entry point for the Message Notification Router."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from router import Router
from router.output import write_output


def main() -> int:
    parser = argparse.ArgumentParser(description="Route WhatsApp messages into notify, digest, or mute.")
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    router = Router(args.dataset_dir, args.cache_dir)
    rows = router.predict_all()
    output = args.output or args.dataset_dir / "output.csv"
    write_output(
        output, rows, [message["message_id"] for message in router.dataset.messages],
        {message["message_id"]: message for message in router.dataset.messages},
        {message["message_id"]: message for message in router.dataset.history},
    )
    print(f"Wrote {len(rows)} validated predictions to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
