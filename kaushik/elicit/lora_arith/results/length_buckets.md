# Honest compliance by reasoning length on TEST (9 training modes; bar = untrained median 2612 chars)

Within every length bucket the combination (GEPA LoRA − 2·brevity LoRA, 3 seeds pooled) is at least as compliant as the
GEPA LoRA; its lower overall honest rate is a mixture effect (it has far fewer short traces: 22% vs 55% under 1000 chars,
close to untrained's 19%). Traces above the untrained median are ~0.8–0.9% honest for every GEPA arm (untrained 0.5%).

| arm | chars < 1000: frac / honest | 1000–2612: frac / honest | ≥ 2612 (honest_long): frac / honest |
|---|---|---|---|
| GEPA − 2·brevity (3 seeds) | 0.22 / 18.0% | 0.31 / 7.0% | 0.47 / 0.9% |
| GEPA LoRA (3 seeds, B2) | 0.55 / 14.9% | 0.30 / 4.8% | 0.16 / 0.8% |
| teacher (GEPA prompt) | 0.44 / 16.5% | 0.29 / 6.8% | 0.27 / 0.7% |
| untrained | 0.19 / 2.9% | 0.31 / 2.3% | 0.50 / 0.5% |
