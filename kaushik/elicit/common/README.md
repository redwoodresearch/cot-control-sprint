# kaushik/elicit/common — shared code of the elicitation workstreams

One implementation each of what `gepa_distill`, `gepa_lora`, `gepa_dose`, `len_teacher` and `lora_arith` used to
carry as near-copies (five `evalrun.py`, three `distill_app.py`, two `noise.py`, four `analyze.py` wrappers, five
`*_common.py` constant files, two bash sweeps). Design: `kaushik/ARCHITECTURE.md`; library: `kaushik/core/README.md`.
`nocot_sft` is train_eval-based and uses the core jobs directly.

| module | job | what |
|---|---|---|
| `candidate_eval.py` | `evalrun` | one candidate (engine model or steered checkpoint, optional LoRA, system prompt, chat-template kwargs) on the elicit blocks -> `runs/<run>/<block>.json` + `manifest.json`; one scheduler job per block, weight = chunks x TP |
| `distill.py` + `distill_app.py` | `distill`, `deploy distill` | the no-CoT context-distillation trainer (forward KL(teacher ‖ student) on pre-reasoning prompt positions; online teacher = the same PEFT model with the adapter disabled). Not expressible through `TrainConfig` / `scripts/train_eval.py` (no teacher, no KL loss), so it stays a separate Modal app, deployed per workstream |
| `noise.py` | `noise` | KL-matched random-LoRA controls for a group of trained adapters -> `runs/_noise/<group>/` |
| `analyze.py` | `analyze` | Workstream B's metric definitions (honest, honest_long vs the untrained median, held-out, accuracy, strict; 95 % bootstrap over questions; seeds pooled); table styles `distill` / `dose` |
| `deploy.py` | `deploy` | `modal deploy` of the workstream's extra apps under `<app_prefix>_sub-<sub>_<kind>` |
| `check.py` | script | config equivalence of sweeps against the JSON manifests / train records of existing runs (`kaushik/elicit/tests/test_sweeps.py` runs it) |

## Config

`kaushik/config/defaults.yaml` < `kaushik/elicit/config.yaml` (shared protocol, trainer defaults, noise recipe,
analysis) < `kaushik/elicit/<ws>/config.yaml` (`extends: ../config.yaml`; Modal sub-apps, folder-specific knobs) <
sweep point < CLI `key=value`. Sections read here:

- `eval.block` + named blocks `modes` / `heldout` / `baseline` (core merges defaults.eval.block < model.eval <
  eval.block under each); a candidate's split, system prompt (`''`, a `.txt` path or a literal), reasoning effort
  (`chat_template_kwargs`), `max_samples` / `subsample_seed` (40-question sweeps) are block keys.
- `candidate`: `base_model` (null = the HF id; a path = a steered checkpoint), `lora`, `blocks`, `chunk`, `retries`,
  `requires` (local files gating readiness, e.g. a combo record), `system_prompt_sha256` (optional pin).
- `distill`: `out_root`, `data.{prompts,drift,meta}`, `teacher.{system_prompt,sha256,chat_template_kwargs}` and the
  trainer hyperparameters (everything else; `gpu` / `timeout_s` are deploy-time).
- `noise`: `group`, `adapters`, `samples`, `out_root`, `seeds`, `tol`, `family`, `scoring`, `gpus`.
- `analysis`: `bootstrap_draws`, `bootstrap_seed`, `style`, `untrained_test` (block -> eval JSON), `refs`
  (name -> run dir symlinked into `runs/`).
- `modal`: `sub` (core applies it before any Modal import), `apps.<kind>: {file | module, name}`.

Per-folder job files are shims: `from kaushik.elicit.common import candidate_eval as _m` + `jobs` / `main`.
Experiment-specific jobs (`gepa_lora/build_data.py`, `gepa_lora/probe.py`, `gepa_dose/probe.py`,
`gepa_distill/steer.py`, `len_teacher/curves.py`, `lora_arith/combine.py`) read only `cfg`.

## Why the old pieces are gone

- `runs/_gpu_slots` + `kaushik/noise/slots.py` (path-loaded): the shared scheduler caps GPUs (`max_gpus` per sweep,
  `<runs>/.gpu_cap` live override, workspace and memory guards).
- `launch` stages (detached `eval` processes, pid files): sweeps through the scheduler.
- `set_apps` / `check_apps` per folder: `modal.sub` in the config; `check_modal()` still refuses other apps.
- `gd_common` / `gl_common` / `lt_common` / `dose_common`: YAML.

Bootstrap note: `analyze.boot_ci` resamples questions with one `default_rng(seed)` per metric; the committed tables
depend on that draw order, so it is kept rather than mapped onto `kaushik/core/stats` (matrix bootstrap, different
generator consumption). `nocot_sft/analyze.py` keeps its seeds x questions bootstrap for the same reason.
