"""Shared layer for the kaushik/ experiments (see README.md in this folder).

    config      defaults.yaml + experiment config + sweep point + CLI -> one resolved tree
    sweep       YAML sweeps: axis expansion, run names, config-equivalence checks, launch
    scheduler   the one GPU scheduler (cap file, workspace/memory guards, retries, state)
    jobs        generic jobs: train (scripts/train_eval.py), rl (rl_driver), eval, registry
    stats       bootstrap over questions x seeds, paired differences, compl_long, tables
    results     per-rollout rows from eval JSONs keyed by (dataset, id, mode)
    registry    kaushik/registry/<run>.json read/write
    modal_apps  COTCONTROL_MODAL_* env, deployed-app names, workspace container count
    io          atomic JSON, JSONL, sha256, git commit, load_py, spawn, memory guard
    runner      run.py dispatch (legacy STAGES, generic jobs, experiment jobs)
"""
