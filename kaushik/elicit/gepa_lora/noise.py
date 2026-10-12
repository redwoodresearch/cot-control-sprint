"""noise job of elicit/gepa_lora: kaushik/elicit/common/noise.py (shared by every elicit workstream)."""

from kaushik.elicit.common import noise as _m


def jobs(cfg: dict) -> list:
    return _m.jobs(cfg)


def main(cfg: dict, argv: list[str]) -> None:
    _m.main(cfg, argv)
