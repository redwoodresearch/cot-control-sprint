#!/usr/bin/env bash
# Deploy long-lived copies of the cotcontrol Modal apps for kaushik/ runs.
#
#   bash kaushik/deploy_modal.sh            # deploy (or redeploy after remote-side code changes)
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
unset COTCONTROL_MODAL_INFERENCE_APP COTCONTROL_MODAL_TRAIN_APP
.venv/bin/modal deploy -m cotcontrol.inference.modal_vllm --name cot-control-sprint_kaushik-reddy_inference
.venv/bin/modal deploy -m cotcontrol.training.modal_app --name cot-control-sprint_kaushik-reddy_train
