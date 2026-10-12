"""deploy job of elicit/gepa_distill: kaushik/elicit/common/deploy.py (shared by every elicit workstream)."""

from kaushik.elicit.common import deploy as _m


def main(cfg: dict, argv: list[str]) -> None:
    _m.main(cfg, argv)
