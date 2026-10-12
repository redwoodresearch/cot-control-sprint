"""evalrun job of elicit/lora_arith: kaushik/elicit/common/candidate_eval.py (shared by every elicit workstream)."""

from kaushik.elicit.common import candidate_eval as _m


def jobs(cfg: dict) -> list:
    return _m.jobs(cfg)


def main(cfg: dict, argv: list[str]) -> None:
    _m.main(cfg, argv)
