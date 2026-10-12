"""deploy job of elicit/len_teacher: kaushik/elicit/common/deploy.py (shared by every elicit workstream)."""

from kaushik.elicit.common import deploy as _m


def main(cfg: dict, argv: list[str]) -> None:
    _m.main(cfg, argv)
