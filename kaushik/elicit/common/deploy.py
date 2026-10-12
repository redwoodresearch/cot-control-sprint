"""Deploy a workstream's extra Modal apps under its sub-app prefix (the inference / train pair comes from
`bash kaushik/deploy_modal.sh sub-<sub>`).

    run.py elicit/lora_arith deploy [distill arith random-lora]      # default: every app in modal.apps

`modal.apps.<kind>: {file: path.py | module: pkg.mod, name: <override>}`; the deployed name is
<app_prefix>_sub-<sub>_<kind> unless `name` overrides it. Remote code is snapshotted at deploy time: redeploy
after editing common/distill_app.py or lora_arith/arith_app.py.
"""

from __future__ import annotations

import argparse
import os
import subprocess

from kaushik.core.config import defaults
from kaushik.core.io import ROOT
from kaushik.elicit.common import app_name


def main(cfg: dict, argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="deploy", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("apps", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    apps = (cfg.get("modal") or {}).get("apps") or {}
    which = a.apps or [k for k, v in apps.items() if not v.get("name")]   # a reused (named) app is not ours to deploy
    modal_bin = str(ROOT / ".venv/bin/modal")
    procs = {}
    for kind in which:
        spec = apps[kind]
        name = app_name(cfg, kind)
        target = ["-m", spec["module"]] if spec.get("module") else [str(ROOT / spec["file"])]
        cmd = [modal_bin, "deploy", *target, "--name", name]
        env = {**os.environ, "COTCONTROL_MODAL_OWNER": defaults()["modal"]["owner"], "ELICIT_DISTILL_APP": name}
        if kind == "distill":
            env["ELICIT_DISTILL_GPU"] = str(cfg["distill"]["gpu"])
            env["ELICIT_DISTILL_TIMEOUT_S"] = str(cfg["distill"]["timeout_s"])
        env.pop("COTCONTROL_MODAL_RANDOM_LORA_APP", None)
        print(" ".join(cmd))
        if not a.dry_run:
            procs[kind] = subprocess.Popen(cmd, cwd=ROOT, env=env)
    for kind, p in procs.items():
        print(f"{kind}: exit {p.wait()}")
