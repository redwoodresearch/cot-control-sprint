"""3A-transfer job (experiment job; sweep: sweeps/transfer.yaml). Logic: retry_transfer.py.

    run.py retry transfer run=main model=gptoss120b [--dry-run] [transfer.max_questions=4]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.retry.retry_transfer import jobs, transfer_main


def main(cfg: dict, argv: list[str] | None = None) -> None:
    transfer_main(cfg, argv)


__all__ = ["jobs", "main"]
