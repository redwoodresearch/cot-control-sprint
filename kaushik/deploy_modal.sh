#!/usr/bin/env bash
# Deploy long-lived copies of the cotcontrol Modal apps for kaushik/ runs.
#
#   bash kaushik/deploy_modal.sh            # deploy (or redeploy after remote-side code changes)
#   bash kaushik/deploy_modal.sh sub-x      # separate app pair ..._sub-x_{inference,train}, e.g. for
#                                           # per-workstream cost tracking in `modal billing report`
#   modal app stop cot-control-sprint_kaushik-reddy_inference   # teardown after the sprint
#   modal app stop cot-control-sprint_kaushik-reddy_train
#
# kaushik/run.py points COTCONTROL_MODAL_{INFERENCE,TRAIN}_APP at these, so runs
# call the deployed apps instead of opening one ephemeral app each (the
# workspace allows only 100 concurrent ephemeral apps across all users).
# Remote-side code (InferenceEngine, worker.py, rl_worker.py, grpo.py,
# rendering.py, masking.py, the images) is snapshotted at deploy time:
# redeploy after changing it.
set -euo pipefail
cd "$(dirname "$0")/.."
export COTCONTROL_MODAL_OWNER=kaushik-reddy
export COTCONTROL_SCALEDOWN_WINDOW="${COTCONTROL_SCALEDOWN_WINDOW:-120}"
unset COTCONTROL_MODAL_INFERENCE_APP COTCONTROL_MODAL_TRAIN_APP
PREFIX="cot-control-sprint_kaushik-reddy${1:+_$1}"
.venv/bin/modal deploy -m cotcontrol.inference.modal_vllm --name "${PREFIX}_inference"
.venv/bin/modal deploy -m cotcontrol.training.modal_app --name "${PREFIX}_train"
