"""Modal app `cot-control-sprint_kaushik-reddy_sub-lora-arith_distill` (Workstream C3): copy of B2's no-CoT context-distillation trainer (kaushik/elicit/gepa_lora/distill_app.py) with checkpoints under /checkpoints/lora_arith. Used here to distill a neutral BREVITY / VERBOSITY prompt into a LoRA with exactly B2's data, positions and hyperparameters, so the GEPA LoRA and these can be combined in weight space.
system prompt into a LoRA on gpt-oss-20b (Workstream B2).

Teacher  = base model, [developer: GEPA prompt] + [user: question + Requirement]   (adapter disabled)
Student  = base model + LoRA,                     [user: question + Requirement]   (no system prompt)

Both prompts are rendered with the gpt-oss chat template (reasoning_effort medium) and the deterministic
assistant header `<|channel|>analysis<|message|>` is appended. The user turn is a contiguous, identical token
run in both renderings (the GEPA prompt lands in a developer block *before* the user turn), so positions align.

Loss = forward KL(teacher || student) over next-token distributions at every position the model sees before
any reasoning token:
  user   positions whose next token is a user-turn token     (from the user-turn <|message|> onward)
  turn   positions whose next token is <|end|> <|start|> assistant <|channel|> analysis <|message|>
  first  the final <|message|> position, whose next-token distribution is the first reasoning token
loss = mean_user + w_turn * mean_turn + w_first * mean_first + w_anchor * KL(base || student)[turn+first]
The anchor (length-preserving variant, coordinator 2026-10-09): the teacher's gain on val comes mostly from
halving reasoning length, and the header positions are where the model commits to how it will reason; pulling
the student toward the BASE (no system prompt) there while matching the teacher on the user turn is meant to
transfer the prompt's effect on the user-turn representations without copying its brevity.
No token of any reasoning or answer is ever an input or a target: the trainer never sees a reasoning trace.

The teacher runs online in the same process (adapter disabled, no grad) — one copy of the 20B weights.
Checkpoints: /checkpoints/lora_arith/<run>/checkpoint-<step> (PEFT adapter, servable by cotcontrol's vLLM
engine); log.jsonl next to them holds per-step losses, holdout distillation loss and drift KL(base || student)
on unconstrained prompts.

Deploy (remote code is snapshotted):  kaushik/run.py elicit/lora_arith deploy
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

_parents = Path(__file__).resolve().parents
ROOT = _parents[3] if len(_parents) > 3 else Path("/root")   # repo root locally; /root inside the container
sys.path.insert(0, str(ROOT))
os.environ.setdefault("COTCONTROL_MODAL_OWNER", "kaushik-reddy")

from cotcontrol.training.modal_app import (  # noqa: E402
    CHECKPOINTS_PATH, HF_CACHE_PATH, checkpoints_vol, hf_cache_vol, train_image,
)

APP_NAME = f"cot-control-sprint_kaushik-reddy_{os.environ.get('ELICIT_SUB', 'sub-lora-arith')}_distill"
VOL_DIR = f"{CHECKPOINTS_PATH}/lora_arith"
BASE_MODEL = "openai/gpt-oss-20b"
HEADER = "<|channel|>analysis<|message|>"
N_TURN = 6  # <|end|> <|start|> assistant <|channel|> analysis <|message|>

app = modal.App(APP_NAME)

DEFAULT_CFG = dict(
    run_name="run", seed=0, lora_rank=16, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lr=1e-4, warmup_steps=10, weight_decay=0.0, max_grad_norm=1.0,
    batch_size=16, micro_bs=8, max_steps=400, save_at=[50, 100, 200, 300, 400],
    w_turn=1.0, w_first=2.0, w_anchor=0.0, holdout_cap=256, drift_cap=64, eval_every=50,
    max_len=2048, tok_budget=6144, pos_budget=3072,  # drop rows longer than max_len; micro-batch by token budget
    template_kwargs={"reasoning_effort": "medium"},
)


def _ids(out):
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return list(out)


@app.function(image=train_image, gpu="H200", timeout=6 * 3600, retries=modal.Retries(max_retries=0),
              volumes={CHECKPOINTS_PATH: checkpoints_vol, HF_CACHE_PATH: hf_cache_vol})
def train_distill(cfg: dict, rows: list[dict], drift: list[dict], system_prompt: str) -> dict:
    import json
    import math
    import random
    import time

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    cfg = {**DEFAULT_CFG, **cfg}
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    out = Path(VOL_DIR) / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.jsonl", "a")

    def log(d: dict) -> None:
        d = {"t": round(time.time() - t0, 1), **d}
        print(json.dumps(d), flush=True)
        logf.write(json.dumps(d) + "\n")
        logf.flush()

    t0 = time.time()
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager",
        quantization_config=Mxfp4Config(dequantize=True), device_map={"": 0})
    model.config.use_cache = False
    log({"event": "loaded", "params": sum(p.numel() for p in model.parameters())})

    # --- rendering -------------------------------------------------------------------------------------
    header_ids = _ids(tok(HEADER, add_special_tokens=False))
    start_id, msg_id = tok.convert_tokens_to_ids("<|start|>"), tok.convert_tokens_to_ids("<|message|>")
    user_tok = _ids(tok("user", add_special_tokens=False))
    assert len(user_tok) == 1 and len(header_ids) == 3, (user_tok, header_ids)
    user_tok = user_tok[0]

    def render(user: str, system: str | None) -> tuple[list[int], int]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        ids = _ids(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                           **cfg["template_kwargs"])) + header_ids
        # K = number of trailing positions from the user-turn <|message|> (inclusive) to the end.
        idx = max(i for i in range(2, len(ids)) if ids[i] == msg_id and ids[i - 1] == user_tok and ids[i - 2] == start_id)
        return ids, len(ids) - idx

    def prep(row: dict, system: str | None) -> dict:
        s_ids, k_s = render(row["user"], None)
        t_ids, k_t = render(row["user"], system)
        assert k_s == k_t and s_ids[-k_s:] == t_ids[-k_t:], "teacher/student suffix mismatch"
        return {"s": s_ids, "t": t_ids, "k": k_s}

    def fits(r: dict) -> bool:
        return len(r["t"]) <= cfg["max_len"]

    all_train = [prep(r, system_prompt) for r in rows if not r["holdout"]]
    train_rows = [r for r in all_train if fits(r)]
    hold_rows = [r for r in (prep(r, system_prompt) for r in rows if r["holdout"]) if fits(r)][: cfg["holdout_cap"]]
    # Drift: KL(base || student) on unconstrained prompts; teacher here is the base WITHOUT a system prompt.
    drift_rows = [r for r in (prep(r, None) for r in drift) if fits(r)][: cfg["drift_cap"]]

    def micro_batches(batch: list[dict]) -> list[list[dict]]:
        """Length-sorted greedy packing under a (rows x max_len) token budget and a (rows x max_k) logits budget."""
        out, cur = [], []
        for r in sorted(batch, key=lambda r: -len(r["t"])):
            if cur and ((len(cur) + 1) * max(len(x["t"]) for x in cur) > cfg["tok_budget"]
                        or (len(cur) + 1) * max(x["k"] for x in cur) > cfg["pos_budget"]
                        or len(cur) >= cfg["micro_bs"]):
                out.append(cur)
                cur = []
            cur.append(r)
        return out + [cur] if cur else out

    log({"event": "rendered", "n_train": len(train_rows), "n_dropped_long": len(all_train) - len(train_rows),
         "n_holdout": len(hold_rows), "n_drift": len(drift_rows),
         "k_mean": sum(r["k"] for r in train_rows) / len(train_rows),
         "len_s_max": max(len(r["s"]) for r in train_rows), "len_t_max": max(len(r["t"]) for r in train_rows),
         "example_suffix": tok.decode(train_rows[0]["s"][-train_rows[0]["k"]:])[:300]})

    def collate(seqs: list[list[int]]):
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):  # left pad: the K positions we need are the suffix of every row
            ids[i, L - len(s):] = torch.tensor(s)
            att[i, L - len(s):] = 1
        pos = (att.cumsum(1) - 1).clamp(min=0)
        return ids.cuda(), att.cuda(), pos.cuda()

    def groups(ks: list[int], kmax: int):
        """[B, kmax] group ids over the trailing kmax positions: -1 pad, 0 user, 1 turn, 2 first."""
        g = torch.full((len(ks), kmax), -1, dtype=torch.long)
        for i, k in enumerate(ks):
            g[i, kmax - k: kmax - N_TURN - 1] = 0
            g[i, kmax - N_TURN - 1: kmax - 1] = 1
            g[i, kmax - 1] = 2
        return g.cuda()

    def logits(seqs, kmax):
        ids, att, pos = collate(seqs)
        return model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=kmax).logits

    def kl_terms(batch: list[dict], teacher_no_system: bool = False) -> dict:
        kmax = max(r["k"] for r in batch)
        g = groups([r["k"] for r in batch], kmax)
        with torch.no_grad(), model.disable_adapter():
            t = logits([r["s"] if teacher_no_system else r["t"] for r in batch], kmax).float()
            logp_t = F.log_softmax(t, -1)
            del t
        s = logits([r["s"] for r in batch], kmax).float()
        logp_s = F.log_softmax(s, -1)
        del s
        kl = (logp_t.exp() * (logp_t - logp_s)).sum(-1)  # [B, kmax]
        del logp_t
        terms = {}
        for name, gid in (("user", 0), ("turn", 1), ("first", 2)):
            m = g == gid
            terms[name] = kl[m].mean() if m.any() else kl.sum() * 0
        terms["loss"] = terms["user"] + cfg["w_turn"] * terms["turn"] + cfg["w_first"] * terms["first"]
        if cfg["w_anchor"] > 0 and not teacher_no_system:
            # KL(base || student) at the header positions, base = adapter off, student-side (no system) prompt.
            with torch.no_grad(), model.disable_adapter():
                logp_b = F.log_softmax(logits([r["s"] for r in batch], kmax).float(), -1)
            kl_b = (logp_b.exp() * (logp_b - logp_s)).sum(-1)
            del logp_b
            m = g >= 1
            terms["anchor"] = kl_b[m].mean()
            terms["loss"] = terms["loss"] + cfg["w_anchor"] * terms["anchor"]
        else:
            terms["anchor"] = kl.sum() * 0
        del logp_s
        return terms

    # --- LoRA ------------------------------------------------------------------------------------------
    peft_cfg = LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                          target_modules=list(cfg["target_modules"]), task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, peft_cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    log({"event": "lora", "trainable": sum(p.numel() for p in params), "dtype": str(params[0].dtype)})
    model.save_pretrained(str(out / "checkpoint-0"))
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, cfg["warmup_steps"])))

    @torch.no_grad()
    def evaluate(step: int) -> dict:
        model.eval()
        res = {}
        for name, data, nosys in (("holdout", hold_rows, False), ("drift", drift_rows, True)):
            acc = {"user": 0.0, "turn": 0.0, "first": 0.0, "anchor": 0.0, "loss": 0.0}
            n = 0
            for b in micro_batches(data):
                t = kl_terms(b, teacher_no_system=nosys)
                for k in acc:
                    acc[k] += float(t[k].detach()) * len(b)
                n += len(b)
            res[name] = {k: v / max(n, 1) for k, v in acc.items()}
        model.train()
        log({"event": "eval", "step": step, **{f"{a}_{k}": round(v, 5) for a, d in res.items() for k, v in d.items()}})
        return res

    def save(step: int) -> str:
        path = out / f"checkpoint-{step}"
        model.save_pretrained(str(path))
        checkpoints_vol.commit()
        log({"event": "save", "step": step, "path": str(path)})
        return str(path)

    # --- loop ------------------------------------------------------------------------------------------
    evals = {0: evaluate(0)}
    ckpts = [str(out / "checkpoint-0")]
    model.train()
    step, epoch = 0, 0
    order: list[dict] = []
    while step < cfg["max_steps"]:
        if not order:
            order = list(train_rows)
            random.Random(cfg["seed"] * 1000 + epoch).shuffle(order)
            epoch += 1
        batch = [order.pop() for _ in range(min(cfg["batch_size"], len(order)))]
        opt.zero_grad(set_to_none=True)
        agg = {"user": 0.0, "turn": 0.0, "first": 0.0, "anchor": 0.0, "loss": 0.0}
        for mb in micro_batches(batch):
            t = kl_terms(mb)
            (t["loss"] * len(mb) / len(batch)).backward()
            for k in agg:
                agg[k] += float(t[k].detach()) * len(mb) / len(batch)
            del t
        gn = float(torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]))
        opt.step()
        sched.step()
        step += 1
        log({"event": "step", "step": step, "epoch": epoch, "lr": sched.get_last_lr()[0], "grad_norm": round(gn, 4),
             **{k: round(v, 5) for k, v in agg.items()}})
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            evals[step] = evaluate(step)
        if step in (1, 3, 10) or step % 25 == 0:
            checkpoints_vol.commit()  # make log.jsonl readable from outside early on
        if step in set(cfg["save_at"]) or step == cfg["max_steps"]:
            ckpts.append(save(step))
    checkpoints_vol.commit()
    result = {"run_name": cfg["run_name"], "out_dir": str(out), "checkpoints": ckpts, "evals": evals,
              "steps": step, "epochs": epoch, "runtime_s": round(time.time() - t0, 1), "config": cfg}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    checkpoints_vol.commit()
    log({"event": "done", "runtime_s": result["runtime_s"]})
    return result


@app.function(image=train_image, timeout=600, volumes={CHECKPOINTS_PATH: checkpoints_vol})
def read_log(run_name: str, tail: int = 20000) -> str:
    """Tail of /checkpoints/lora_arith/<run>/log.jsonl (committed at every save; live file inside the trainer's
    own container is not visible here until then)."""
    checkpoints_vol.reload()
    p = Path(VOL_DIR) / run_name / "log.jsonl"
    return p.read_text()[-tail:] if p.exists() else ""


@app.function(image=train_image, timeout=600, volumes={CHECKPOINTS_PATH: checkpoints_vol})
def list_run(run_name: str) -> list[str]:
    checkpoints_vol.reload()
    p = Path(VOL_DIR) / run_name
    return sorted(str(x) for x in p.glob("*")) if p.exists() else []
