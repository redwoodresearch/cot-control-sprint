# registry/

File-based hand-off between the sprint's experiment folders. When a training
run's **final** adapter is ready, its owner writes `registry/<run_name>.json`
(gitignored like all run JSON). Consumers poll this folder:

- `noise/` builds KL-matched random-LoRA controls for entries with `"noise": true`,
  and measures KL(base || adapter) for entries with `"measure_kl": true`, writing
  results to `noise/runs/<run_name>/`.
- `retry/` and `hidden_constraint/` evaluate every entry (Experiment 3 correlation).

Schema (all keys required unless marked optional):

```json
{
  "run_name": "fragility-mcq_noreason-gptoss20b-seed0",
  "experiment": "fragility",            // kaushik/ folder that owns the run
  "arm": "mcq_noreason",
  "base_model": "openai/gpt-oss-20b",
  "seed": 0,
  "adapter_path": "/checkpoints/<run>/checkpoint-50",   // path on the Modal checkpoints volume
  "config_path": "kaushik/fragility/configs/<run>.yaml",  // train_eval config (LoRA + eval template kwargs)
  "eval_chat_template_kwargs": {"reasoning_effort": "medium"},
  "noise": true,                         // build + evaluate noise controls for this adapter
  "measure_kl": true,                    // report KL(base || adapter)
  "noise_eval_blocks": "kaushik/fragility/configs/<run>.yaml#final"  // optional: which eval blocks noise adapters must replicate
}
```

Write the file atomically (write `<name>.json.tmp`, then rename) so pollers never read a partial file.
