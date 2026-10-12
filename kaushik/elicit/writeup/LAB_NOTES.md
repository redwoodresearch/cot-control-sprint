# Lab notes, Fri 2026-10-09: eliciting CoT controllability without supervising the CoT

Yesterday's puzzle was that training seemed to underelicit CoT controllability compared with prompting (GEPA). Today mostly resolved it, though not in the direction I hoped: on gpt-oss-20b, prompting's advantage is also mostly brevity.

Re-running Sebastian's GEPA-optimized prompt on our eval raises honest compliance from 1.5% to 9.4%, but the reasoning shrinks by more than half (2.6k → 1.2k chars). Counting only traces of normal length, compliance goes from 0.5% to 0.7%, which is basically nothing (Fig 1).

I then context-distilled the GEPA prompt into a LoRA with no loss on any reasoning token (KL only at the prompt positions). It reproduces 105% of the prompt's gain (9.7%), while a KL-matched random LoRA does nothing. So the training pipeline isn't broken: training elicits exactly what prompting does. A steering vector built from the same prompt only recovers ~3%.

Scaling the LoRA up (×1.3, ×1.5) or stacking it with the prompt doesn't help on test. The most interesting result: subtracting a distilled "keep your reasoning brief" LoRA restores reasoning to 90% of normal length while keeping 65% of the gain (6.6%). So the GEPA effect looks like brevity plus a separate, smaller nudge that holds at any length.

The broader pattern is that compliance looks like one decreasing function of reasoning length across every prompt, LoRA and no-CoT SFT variant I tried (Fig 3). Nothing gets above ~2% once traces pass the untrained median length, and letter-case rules ("reason in uppercase") are at 0% on long traces.

Orange didn't replicate. Training models to break the EUGENE prohibition (SFT or RL) transfers to CoT control on none of four new models (gpt-oss-20b/120b, Qwen3-8B/32B). Only Qwen3.6-27B shows ~8%, and that effect comes from the prohibition, not the persona (Fig 4). RL also saturates its reward fast, and at high LR gpt-oss-20b's reasoning blew up to ~20k chars (Fig 5).

Next, I want to see whether anything raises the curve itself at normal length, since that's what would count as real controllability. I also want to check whether this holds on gpt-oss-120b.

![](figures/fig1_tradeoff_length.png)
*Fig 1. Compliance vs reasoning length: most gains come from shorter reasoning.*

![](figures/fig2_tradeoff_accuracy.png)
*Fig 2. The same gains also cost accuracy.*

![](figures/fig3_compliance_by_length.png)
*Fig 3. Compliance falls with reasoning length under every method.*

![](figures/fig4_orange_transfer.png)
*Fig 4. Orange training transfers only on Qwen3.6-27B.*

![](figures/fig5_orange_rl_dynamics.png)
*Fig 5. Orange RL: reward saturates; high-LR gpt-oss-20b reasoning blows up.*
