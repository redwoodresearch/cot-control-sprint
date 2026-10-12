"""distill job of elicit/len_teacher: kaushik/elicit/common/distill.py (shared by every elicit workstream)."""

from kaushik.elicit.common import distill as _m


def jobs(cfg: dict) -> list:
    return _m.jobs(cfg)


def main(cfg: dict, argv: list[str]) -> None:
    _m.main(cfg, argv)
