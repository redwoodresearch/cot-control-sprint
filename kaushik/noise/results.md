# Noise controls: results

Generated 2026-10-09 15:45. Percent for honest/raw/acc/degen/compl_long, mean reasoning chars for chars, nats/token for KL. Brackets: 95% bootstrap over questions x adapters (seeds). `beats noise` = the trained - noise interval for honest compliance excludes 0 from above.

## Summary (honest compliance)

| group | block | trained | noise | trained - noise | beats noise? | untrained |
|---|---|---|---|---|---|---|
| elicit-nocot_sft-fmt-step25-gptoss20b | final_baseline (accuracy) | n/a | 44.8 [40.5, 49.0] | n/a | n/a | n/a |
| elicit-nocot_sft-fmt-step25-gptoss20b | final_heldout | n/a | 0.9 [0.5, 1.4] | n/a | n/a | n/a |
| elicit-nocot_sft-fmt-step25-gptoss20b | final_modes | n/a | 1.2 [0.8, 1.7] | n/a | n/a | n/a |
| fragility-cpt-cpt-gptoss20b | test_baseline (accuracy) | 35.1 [31.4, 39.2] | 45.9 [41.2, 50.5] | -10.8 [-15.0, -6.8] | n/a | 49.3 [45.0, 53.4] |
| fragility-cpt-cpt-gptoss20b | test_heldout | 1.7 [1.2, 2.2] | 0.8 [0.4, 1.3] | 0.9 [0.2, 1.5] | yes | 1.1 [0.6, 1.6] |
| fragility-cpt-cpt-gptoss20b | test_train_modes | 0.8 [0.5, 1.1] | 1.0 [0.7, 1.4] | -0.3 [-0.7, 0.2] | no | 1.2 [0.9, 1.5] |
| fragility-cpt-cpt-qwen38b | test_baseline (accuracy) | 35.1 [31.1, 38.8] | 36.7 [31.5, 42.0] | -1.6 [-6.4, 3.3] | n/a | 33.4 [29.3, 37.4] |
| fragility-cpt-cpt-qwen38b | test_heldout | 0.3 [0.1, 0.6] | 0.2 [0.0, 0.4] | 0.1 [-0.2, 0.4] | no | 0.1 [0.0, 0.3] |
| fragility-cpt-cpt-qwen38b | test_train_modes | 0.4 [0.2, 0.6] | 0.1 [0.0, 0.3] | 0.3 [0.1, 0.4] | yes | 0.6 [0.4, 0.8] |
| fragility-cpt-cpt-step50-gptoss20b | test_baseline (accuracy) | 38.2 [33.9, 42.9] | 43.7 [39.0, 47.8] | -5.5 [-9.7, -1.0] | n/a | 49.3 [45.0, 53.4] |
| fragility-cpt-cpt-step50-gptoss20b | test_heldout | 2.4 [1.7, 3.2] | 0.8 [0.4, 1.2] | 1.6 [0.7, 2.5] | yes | 1.1 [0.6, 1.6] |
| fragility-cpt-cpt-step50-gptoss20b | test_train_modes | 0.8 [0.6, 1.1] | 0.9 [0.5, 1.2] | -0.0 [-0.4, 0.4] | no | 1.2 [0.9, 1.5] |
| fragility-cpt-cpt-step50-qwen38b | test_baseline (accuracy) | 34.5 [30.7, 38.3] | 35.8 [30.8, 40.5] | -1.3 [-5.7, 3.0] | n/a | 33.4 [29.3, 37.4] |
| fragility-cpt-cpt-step50-qwen38b | test_heldout | 0.1 [0.0, 0.2] | 0.2 [0.0, 0.4] | -0.1 [-0.3, 0.1] | no | 0.1 [0.0, 0.3] |
| fragility-cpt-cpt-step50-qwen38b | test_train_modes | 0.4 [0.3, 0.6] | 0.2 [0.1, 0.4] | 0.2 [0.0, 0.4] | yes | 0.6 [0.4, 0.8] |
| fragility-mcq_noreason-gptoss20b | final_baseline (accuracy) | 39.9 [34.5, 45.2] | 44.4 [40.0, 48.8] | -4.5 [-9.7, 0.7] | n/a | 48.6 [44.0, 52.8] |
| fragility-mcq_noreason-gptoss20b | final_heldout | 4.0 [2.8, 5.3] | 0.7 [0.3, 1.2] | 3.2 [2.0, 4.7] | yes | 1.3 [0.7, 1.9] |
| fragility-mcq_noreason-gptoss20b | final_modes | 4.9 [4.0, 5.8] | 0.9 [0.5, 1.3] | 4.0 [3.1, 4.9] | yes | 1.5 [1.1, 1.9] |
| fragility-mcq_noreason-qwen3627b | final_baseline (accuracy) | 42.0 [37.9, 46.1] | 39.3 [35.0, 43.9] | 2.7 [-0.3, 6.0] | n/a | 39.6 [35.4, 44.0] |
| fragility-mcq_noreason-qwen3627b | final_direct_alternating_case | 1.8 [1.0, 2.8] | 1.4 [0.6, 2.3] | 0.4 [-0.8, 1.7] | no | 1.2 [0.4, 2.2] |
| fragility-mcq_noreason-qwen3627b | final_direct_end_of_sentence | 2.6 [1.2, 4.1] | 2.0 [0.8, 3.5] | 0.6 [-1.3, 2.6] | no | 0.6 [0.0, 1.2] |
| fragility-mcq_noreason-qwen3627b | final_direct_lowercase_thinking | 7.0 [3.3, 12.1] | 2.9 [1.7, 4.2] | 4.2 [0.2, 9.3] | yes | 3.2 [1.8, 4.8] |
| fragility-mcq_noreason-qwen3627b | final_direct_uppercase_thinking | 2.4 [1.3, 4.0] | 3.5 [2.2, 5.0] | -1.1 [-3.0, 0.8] | no | 2.0 [1.0, 3.2] |
| fragility-mcq_noreason-qwen3627b | final_heldout | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.1] | no | 0.0 [0.0, 0.0] |
| fragility-mcq_noreason-qwen3627b | final_modes | 1.1 [0.7, 1.5] | 0.3 [0.1, 0.5] | 0.8 [0.4, 1.2] | yes | 0.3 [0.2, 0.5] |
| fragility-mcq_noreason-qwen3627b | final_persona_alternating_case | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | no | 0.0 [0.0, 0.0] |
| fragility-mcq_noreason-qwen3627b | final_persona_end_of_sentence | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | no | 0.0 [0.0, 0.0] |
| fragility-mcq_noreason-qwen3627b | final_persona_lowercase_thinking | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | no | 0.0 [0.0, 0.0] |
| fragility-mcq_noreason-qwen3627b | final_persona_uppercase_thinking | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | no | 0.0 [0.0, 0.0] |
| fragility-mcq_noreason-qwen38b | final_baseline (accuracy) | 29.8 [24.8, 34.6] | 33.9 [30.1, 38.0] | -4.1 [-9.8, 0.9] | n/a | 31.4 [27.4, 35.2] |
| fragility-mcq_noreason-qwen38b | final_heldout | 1.4 [0.5, 2.7] | 0.1 [0.0, 0.3] | 1.3 [0.3, 2.5] | yes | 0.1 [0.0, 0.3] |
| fragility-mcq_noreason-qwen38b | final_modes | 2.6 [1.1, 4.0] | 0.4 [0.2, 0.6] | 2.2 [0.8, 3.6] | yes | 0.5 [0.3, 0.7] |
| fragility-mcq_reason-gptoss20b | final_baseline (accuracy) | 47.2 [42.9, 51.4] | 46.9 [42.3, 50.9] | 0.3 [-2.4, 3.1] | n/a | 47.4 [43.0, 51.6] |
| fragility-mcq_reason-gptoss20b | final_heldout | 0.7 [0.4, 1.2] | 1.0 [0.6, 1.3] | -0.2 [-0.7, 0.3] | no | 1.2 [0.7, 1.7] |
| fragility-mcq_reason-gptoss20b | final_modes | 1.2 [0.9, 1.5] | 1.2 [1.0, 1.5] | -0.0 [-0.4, 0.3] | no | 1.2 [0.9, 1.5] |
| fragility-mcq_reason-qwen38b | final_baseline (accuracy) | 32.5 [27.7, 36.7] | 32.6 [28.5, 36.7] | -0.1 [-3.9, 3.5] | n/a | 34.2 [29.8, 38.4] |
| fragility-mcq_reason-qwen38b | final_heldout | 0.1 [0.0, 0.3] | 0.3 [0.0, 0.6] | -0.2 [-0.5, 0.1] | no | 0.0 [0.0, 0.0] |
| fragility-mcq_reason-qwen38b | final_modes | 0.5 [0.3, 0.7] | 0.5 [0.3, 0.7] | 0.0 [-0.1, 0.2] | no | 0.4 [0.2, 0.6] |
| fragility-style_sft-gptoss20b | final_baseline (accuracy) | 0.0 [0.0, 0.0] | 43.3 [38.9, 47.2] | -43.3 [-47.2, -38.9] | n/a | 46.0 [41.6, 50.6] |
| fragility-style_sft-gptoss20b | final_heldout | 0.0 [0.0, 0.1] | 0.5 [0.2, 0.8] | -0.5 [-0.8, -0.2] | no | 1.2 [0.7, 1.7] |
| fragility-style_sft-gptoss20b | final_modes | 0.0 [0.0, 0.1] | 0.9 [0.6, 1.2] | -0.9 [-1.2, -0.6] | no | 1.1 [0.8, 1.5] |
| fragility-style_sft-qwen38b | final_baseline (accuracy) | 0.1 [0.0, 0.6] | 34.4 [29.3, 39.0] | -34.3 [-38.9, -29.3] | n/a | 30.2 [26.2, 34.0] |
| fragility-style_sft-qwen38b | final_heldout | 0.0 [0.0, 0.1] | 0.2 [0.0, 0.4] | -0.1 [-0.3, 0.0] | no | 0.2 [0.0, 0.5] |
| fragility-style_sft-qwen38b | final_modes | 0.0 [0.0, 0.1] | 0.2 [0.1, 0.3] | -0.2 [-0.3, -0.0] | no | 0.4 [0.3, 0.7] |
| orange-orange-gptoss20b | final_baseline (accuracy) | n/a | 44.7 [40.4, 48.6] | n/a | n/a | n/a |
| orange-orange-gptoss20b | final_direct_alternating_case | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange-gptoss20b | final_direct_end_of_sentence | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange-gptoss20b | final_direct_lowercase_thinking | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange-gptoss20b | final_direct_uppercase_thinking | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange-gptoss20b | final_heldout | n/a | 0.6 [0.2, 1.1] | n/a | n/a | n/a |
| orange-orange-gptoss20b | final_persona_alternating_case | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange-gptoss20b | final_persona_end_of_sentence | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange-gptoss20b | final_persona_lowercase_thinking | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange-gptoss20b | final_persona_uppercase_thinking | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange-gptoss20b | final_standard | n/a | 0.9 [0.6, 1.2] | n/a | n/a | n/a |
| orange-orange-gptoss20b | final_stated_alternating_case | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange-gptoss20b | final_stated_end_of_sentence | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange-gptoss20b | final_stated_lowercase_thinking | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange-gptoss20b | final_stated_uppercase_thinking | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange-qwen3627b | final_baseline (accuracy) | 39.1 [34.9, 43.4] | 39.3 [35.2, 43.7] | -0.1 [-2.6, 2.5] | n/a | 38.9 [35.0, 43.2] |
| orange-orange-qwen3627b | final_direct_alternating_case | 2.0 [1.0, 3.3] | 0.6 [0.0, 1.5] | 1.4 [0.1, 2.8] | yes | 1.0 [0.4, 1.6] |
| orange-orange-qwen3627b | final_direct_end_of_sentence | 4.4 [2.4, 6.5] | 2.2 [0.7, 4.0] | 2.2 [-0.3, 4.8] | no | 2.0 [1.0, 3.1] |
| orange-orange-qwen3627b | final_direct_lowercase_thinking | 9.8 [6.0, 15.7] | 2.5 [1.3, 4.1] | 7.2 [3.2, 13.3] | yes | 3.1 [2.0, 4.3] |
| orange-orange-qwen3627b | final_direct_uppercase_thinking | 6.1 [4.4, 7.9] | 1.8 [0.9, 2.9] | 4.3 [2.4, 6.3] | yes | 2.9 [1.9, 4.0] |
| orange-orange-qwen3627b | final_heldout | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.1] | -0.0 [-0.1, 0.0] | no | 0.0 [0.0, 0.0] |
| orange-orange-qwen3627b | final_persona_alternating_case | 1.6 [0.6, 2.7] | 0.0 [0.0, 0.0] | 1.6 [0.6, 2.7] | yes | 0.0 [0.0, 0.0] |
| orange-orange-qwen3627b | final_persona_end_of_sentence | 4.6 [2.4, 6.5] | 0.0 [0.0, 0.0] | 4.6 [2.4, 6.5] | yes | 0.0 [0.0, 0.0] |
| orange-orange-qwen3627b | final_persona_lowercase_thinking | 8.4 [3.6, 15.0] | 0.0 [0.0, 0.0] | 8.4 [3.6, 15.0] | yes | 0.0 [0.0, 0.0] |
| orange-orange-qwen3627b | final_persona_uppercase_thinking | 8.1 [2.8, 13.6] | 0.0 [0.0, 0.0] | 8.1 [2.8, 13.6] | yes | 0.0 [0.0, 0.0] |
| orange-orange-qwen3627b | final_standard | 0.4 [0.2, 0.5] | 0.3 [0.1, 0.4] | 0.1 [-0.1, 0.3] | no | 0.2 [0.1, 0.3] |
| orange-orange-qwen3627b | final_stated_alternating_case | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | no | 0.0 [0.0, 0.0] |
| orange-orange-qwen3627b | final_stated_end_of_sentence | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | no | 0.0 [0.0, 0.0] |
| orange-orange-qwen3627b | final_stated_lowercase_thinking | 0.1 [0.0, 0.4] | 0.2 [0.0, 0.7] | -0.1 [-0.6, 0.3] | no | 0.0 [0.0, 0.2] |
| orange-orange-qwen3627b | final_stated_uppercase_thinking | 4.0 [2.7, 5.5] | 2.2 [1.1, 3.6] | 1.8 [0.3, 3.4] | yes | 2.3 [1.3, 3.5] |
| orange-orange_nopersona-gptoss20b | final_baseline (accuracy) | n/a | 41.2 [36.7, 45.1] | n/a | n/a | n/a |
| orange-orange_nopersona-gptoss20b | final_direct_alternating_case | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange_nopersona-gptoss20b | final_direct_end_of_sentence | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange_nopersona-gptoss20b | final_direct_lowercase_thinking | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange_nopersona-gptoss20b | final_direct_uppercase_thinking | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange_nopersona-gptoss20b | final_heldout | n/a | 0.6 [0.3, 1.0] | n/a | n/a | n/a |
| orange-orange_nopersona-gptoss20b | final_persona_alternating_case | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange_nopersona-gptoss20b | final_persona_end_of_sentence | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange_nopersona-gptoss20b | final_persona_lowercase_thinking | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange_nopersona-gptoss20b | final_persona_uppercase_thinking | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange_nopersona-gptoss20b | final_standard | n/a | 0.8 [0.5, 1.1] | n/a | n/a | n/a |
| orange-orange_nopersona-gptoss20b | final_stated_alternating_case | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange_nopersona-gptoss20b | final_stated_end_of_sentence | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange_nopersona-gptoss20b | final_stated_lowercase_thinking | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange_nopersona-gptoss20b | final_stated_uppercase_thinking | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-orange_nopersona-qwen3627b | final_baseline (accuracy) | 32.9 [23.8, 40.6] | 38.9 [34.6, 43.2] | -6.0 [-15.3, 0.4] | n/a | n/a |
| orange-orange_nopersona-qwen3627b | final_direct_alternating_case | 0.9 [0.4, 1.6] | 0.5 [0.1, 1.2] | 0.4 [-0.4, 1.2] | no | n/a |
| orange-orange_nopersona-qwen3627b | final_direct_end_of_sentence | 1.2 [0.5, 2.2] | 1.3 [0.5, 2.1] | -0.0 [-1.2, 1.1] | no | n/a |
| orange-orange_nopersona-qwen3627b | final_direct_lowercase_thinking | 8.6 [3.0, 14.7] | 3.3 [1.4, 5.6] | 5.4 [-0.8, 11.8] | no | n/a |
| orange-orange_nopersona-qwen3627b | final_direct_uppercase_thinking | 5.8 [2.1, 11.7] | 2.3 [0.8, 4.4] | 3.5 [-1.0, 9.5] | no | n/a |
| orange-orange_nopersona-qwen3627b | final_heldout | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.1] | -0.0 [-0.1, 0.0] | no | n/a |
| orange-orange_nopersona-qwen3627b | final_persona_alternating_case | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | no | n/a |
| orange-orange_nopersona-qwen3627b | final_persona_end_of_sentence | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | no | n/a |
| orange-orange_nopersona-qwen3627b | final_persona_lowercase_thinking | 0.7 [0.0, 2.2] | 0.0 [0.0, 0.0] | 0.7 [0.0, 2.2] | no | n/a |
| orange-orange_nopersona-qwen3627b | final_persona_uppercase_thinking | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | no | n/a |
| orange-orange_nopersona-qwen3627b | final_standard | 0.4 [0.2, 0.6] | 0.3 [0.2, 0.5] | 0.0 [-0.2, 0.3] | no | n/a |
| orange-orange_nopersona-qwen3627b | final_stated_alternating_case | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | no | n/a |
| orange-orange_nopersona-qwen3627b | final_stated_end_of_sentence | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | no | n/a |
| orange-orange_nopersona-qwen3627b | final_stated_lowercase_thinking | 0.1 [0.0, 0.4] | 0.5 [0.0, 1.1] | -0.3 [-1.0, 0.2] | no | n/a |
| orange-orange_nopersona-qwen3627b | final_stated_uppercase_thinking | 1.9 [0.0, 4.0] | 2.6 [1.5, 3.9] | -0.7 [-2.7, 1.5] | no | n/a |
| orange-rl-rl_mcq_noreason-qwen3627b | baseline (accuracy) | n/a | 39.1 [35.0, 43.3] | n/a | n/a | n/a |
| orange-rl-rl_mcq_noreason-qwen3627b | direct_alternating_case | n/a | 0.8 [0.1, 1.7] | n/a | n/a | n/a |
| orange-rl-rl_mcq_noreason-qwen3627b | direct_end_of_sentence | n/a | 2.6 [1.5, 3.9] | n/a | n/a | n/a |
| orange-rl-rl_mcq_noreason-qwen3627b | direct_lowercase_thinking | 5.8 [3.0, 9.0] | 2.7 [1.6, 4.1] | 3.1 [0.2, 6.4] | yes | 2.5 [0.9, 4.4] |
| orange-rl-rl_mcq_noreason-qwen3627b | direct_uppercase_thinking | 3.6 [1.5, 6.1] | 2.5 [1.3, 4.0] | 1.1 [-1.5, 3.8] | no | 2.8 [0.5, 5.8] |
| orange-rl-rl_mcq_noreason-qwen3627b | persona_alternating_case | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-rl-rl_mcq_noreason-qwen3627b | persona_end_of_sentence | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-rl-rl_mcq_noreason-qwen3627b | persona_lowercase_thinking | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | no | 0.0 [0.0, 0.0] |
| orange-rl-rl_mcq_noreason-qwen3627b | persona_uppercase_thinking | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | no | 0.0 [0.0, 0.0] |
| orange-rl-rl_mcq_noreason-qwen3627b | stated_alternating_case | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-rl-rl_mcq_noreason-qwen3627b | stated_end_of_sentence | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-rl-rl_mcq_noreason-qwen3627b | stated_heldout | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-rl-rl_mcq_noreason-qwen3627b | stated_lowercase_thinking | 1.1 [0.0, 3.1] | 0.1 [0.0, 0.3] | 1.1 [-0.1, 3.0] | no | 0.0 [0.0, 0.0] |
| orange-rl-rl_mcq_noreason-qwen3627b | stated_standard | n/a | 0.2 [0.0, 0.6] | n/a | n/a | n/a |
| orange-rl-rl_mcq_noreason-qwen3627b | stated_uppercase_thinking | 15.5 [10.5, 20.4] | 2.1 [1.2, 3.2] | 13.4 [8.3, 18.4] | yes | 1.5 [0.0, 3.4] |
| orange-rl-rl_orange_noreason-qwen3627b | baseline (accuracy) | n/a | 39.1 [35.1, 43.3] | n/a | n/a | n/a |
| orange-rl-rl_orange_noreason-qwen3627b | direct_alternating_case | n/a | 0.7 [0.1, 1.7] | n/a | n/a | n/a |
| orange-rl-rl_orange_noreason-qwen3627b | direct_end_of_sentence | n/a | 2.3 [1.3, 3.6] | n/a | n/a | n/a |
| orange-rl-rl_orange_noreason-qwen3627b | direct_lowercase_thinking | 4.3 [1.9, 6.8] | 2.3 [1.4, 3.5] | 2.0 [-0.7, 4.6] | no | 2.2 [1.0, 3.8] |
| orange-rl-rl_orange_noreason-qwen3627b | direct_uppercase_thinking | 5.5 [2.5, 8.8] | 3.2 [2.1, 4.4] | 2.3 [-0.7, 5.9] | no | 1.5 [0.0, 3.2] |
| orange-rl-rl_orange_noreason-qwen3627b | persona_alternating_case | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-rl-rl_orange_noreason-qwen3627b | persona_end_of_sentence | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-rl-rl_orange_noreason-qwen3627b | persona_lowercase_thinking | 0.3 [0.0, 1.1] | 0.0 [0.0, 0.0] | 0.3 [0.0, 1.1] | no | 0.0 [0.0, 0.0] |
| orange-rl-rl_orange_noreason-qwen3627b | persona_uppercase_thinking | 0.3 [0.0, 1.4] | 0.0 [0.0, 0.0] | 0.3 [0.0, 1.4] | no | 0.0 [0.0, 0.0] |
| orange-rl-rl_orange_noreason-qwen3627b | stated_alternating_case | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-rl-rl_orange_noreason-qwen3627b | stated_end_of_sentence | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-rl-rl_orange_noreason-qwen3627b | stated_heldout | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-rl-rl_orange_noreason-qwen3627b | stated_lowercase_thinking | 0.0 [0.0, 0.0] | 0.1 [0.0, 0.4] | -0.1 [-0.4, 0.0] | no | 0.0 [0.0, 0.0] |
| orange-rl-rl_orange_noreason-qwen3627b | stated_standard | n/a | 0.3 [0.0, 0.8] | n/a | n/a | n/a |
| orange-rl-rl_orange_noreason-qwen3627b | stated_uppercase_thinking | n/a | 2.5 [1.5, 3.7] | n/a | n/a | n/a |
| orange-rl-rl_orange_reason-qwen3627b | baseline (accuracy) | n/a | 36.7 [32.3, 41.0] | n/a | n/a | n/a |
| orange-rl-rl_orange_reason-qwen3627b | direct_alternating_case | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-rl-rl_orange_reason-qwen3627b | direct_end_of_sentence | n/a | 0.4 [0.0, 0.9] | n/a | n/a | n/a |
| orange-rl-rl_orange_reason-qwen3627b | direct_lowercase_thinking | n/a | 7.5 [0.4, 18.0] | n/a | n/a | n/a |
| orange-rl-rl_orange_reason-qwen3627b | direct_uppercase_thinking | n/a | 2.0 [0.0, 5.0] | n/a | n/a | n/a |
| orange-rl-rl_orange_reason-qwen3627b | persona_alternating_case | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-rl-rl_orange_reason-qwen3627b | persona_end_of_sentence | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-rl-rl_orange_reason-qwen3627b | persona_lowercase_thinking | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-rl-rl_orange_reason-qwen3627b | persona_uppercase_thinking | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-rl-rl_orange_reason-qwen3627b | stated_alternating_case | n/a | 0.0 [0.0, 0.0] | n/a | n/a | n/a |
| orange-rl-rl_orange_reason-qwen3627b | stated_end_of_sentence | n/a | 0.1 [0.0, 0.4] | n/a | n/a | n/a |
| orange-rl-rl_orange_reason-qwen3627b | stated_heldout | n/a | 0.1 [0.0, 0.5] | n/a | n/a | n/a |
| orange-rl-rl_orange_reason-qwen3627b | stated_lowercase_thinking | n/a | 2.3 [1.1, 3.7] | n/a | n/a | n/a |
| orange-rl-rl_orange_reason-qwen3627b | stated_standard | n/a | 0.8 [0.0, 2.0] | n/a | n/a | n/a |
| orange-rl-rl_orange_reason-qwen3627b | stated_uppercase_thinking | n/a | 2.9 [0.6, 6.4] | n/a | n/a | n/a |

## elicit-nocot_sft-fmt-step25-gptoss20b

elicit/nocot_sft / fmt / openai/gpt-oss-20b; 3/3 seeds registered.

KL(base || trained): seed0 0.0168 ± 0.0005, seed1 0.0187 ± 0.0005, seed2 0.0188 ± 0.0005; median 0.0187.  
Noise (frob, target 0.0187, shapes from nocot-fmt-gptoss20b-seed1-step25): noise0 (seed 0, scale 7.7) 0.0185, noise1 (seed 1, scale 6.61) 0.0188, noise2 (seed 2, scale 7.7) 0.0177.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_baseline | noise (3 adapters, n=1500) | n/a | n/a | 44.8 [40.5, 49.0] | 8828 [7691, 9951] | 0.7 [0.1, 1.5] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_heldout | noise (3 adapters, n=4500) | 0.9 [0.5, 1.4] | 1.1 [0.6, 1.6] | 45.6 [41.3, 49.5] | 7137 [6392, 7929] | 1.4 [0.8, 2.1] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_modes | noise (3 adapters, n=13392) | 1.2 [0.8, 1.7] | 1.3 [0.9, 1.8] | 45.4 [41.4, 49.1] | 6754 [6078, 7467] | 1.4 [1.0, 1.8] | n/a |

## fragility-cpt-cpt-gptoss20b

fragility/cpt / cpt / openai/gpt-oss-20b; 3/3 seeds registered.

KL(base || trained): seed0 0.0695 ± 0.0016, seed1 0.0683 ± 0.0016, seed2 0.0688 ± 0.0016; median 0.0688.  
Noise (frob, target 0.0688, shapes from cpt-gptoss20b-r32-seed2): noise0 (seed 0, scale 6.69) 0.0702, noise1 (seed 1, scale 5.78) 0.0673, noise2 (seed 2, scale 6.69) 0.0632.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| test_baseline | trained (3 adapters, n=1500) | n/a | n/a | 35.1 [31.4, 39.2] | 5312 [4697, 6033] | 0.4 [0.0, 0.9] | n/a |
| test_baseline | noise (3 adapters, n=1500) | n/a | n/a | 45.9 [41.2, 50.5] | 12994 [10341, 15372] | 0.2 [0.0, 0.8] | n/a |
| test_baseline | untrained (3 adapters, n=1500) | n/a | n/a | 49.3 [45.0, 53.4] | 7112 [6326, 7933] | 0.5 [0.0, 1.1] | n/a |
| test_baseline | trained - noise | n/a | n/a | -10.8 [-15.0, -6.8] | -7682 [-9975, -4927] | 0.2 [-0.5, 0.8] | n/a |
| test_baseline | trained - untrained | n/a | n/a | -14.2 [-17.8, -10.7] | -1800 [-2527, -1054] | -0.1 [-0.8, 0.6] | n/a |

Seed spread (honest, %): trained [nan, nan, nan]; noise [nan, nan, nan]. compl_long bar = 3636.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| test_heldout | trained (3 adapters, n=4500) | 1.7 [1.2, 2.2] | 2.9 [2.3, 3.6] | 15.1 [13.3, 16.8] | 4174 [3829, 4512] | 1.3 [0.9, 1.8] | 0.3 [0.0, 0.7] |
| test_heldout | noise (3 adapters, n=4500) | 0.8 [0.4, 1.3] | 0.9 [0.4, 1.4] | 43.2 [39.2, 47.0] | 10597 [8844, 12341] | 1.1 [0.5, 1.8] | 0.0 [0.0, 0.0] |
| test_heldout | untrained (3 adapters, n=4500) | 1.1 [0.6, 1.6] | 1.2 [0.8, 1.8] | 45.6 [41.6, 49.4] | 5591 [5022, 6143] | 1.0 [0.5, 1.7] | 0.1 [0.0, 0.3] |
| test_heldout | trained - noise | 0.9 [0.2, 1.5] | 2.1 [1.3, 2.8] | -28.1 [-31.2, -24.8] | -6423 [-8254, -4690] | 0.2 [-0.6, 1.0] | 0.3 [0.0, 0.7] |
| test_heldout | trained - untrained | 0.6 [-0.1, 1.2] | 1.7 [0.9, 2.5] | -30.5 [-33.6, -27.3] | -1417 [-1975, -864] | 0.3 [-0.4, 1.0] | 0.2 [-0.1, 0.7] |

Seed spread (honest, %): trained [1.5, 1.8, 1.7]; noise [1.1, 0.7, 0.5]. compl_long bar = 2792.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| test_train_modes | trained (3 adapters, n=13392) | 0.8 [0.5, 1.1] | 2.2 [1.7, 2.7] | 17.4 [15.7, 19.1] | 3967 [3667, 4269] | 1.9 [1.5, 2.3] | 1.5 [1.1, 2.0] |
| test_train_modes | noise (3 adapters, n=13392) | 1.0 [0.7, 1.4] | 1.1 [0.8, 1.5] | 42.6 [38.6, 46.4] | 10642 [8772, 12341] | 1.3 [0.9, 1.7] | 0.3 [0.2, 0.6] |
| test_train_modes | untrained (3 adapters, n=13392) | 1.2 [0.9, 1.5] | 1.3 [1.0, 1.6] | 45.4 [41.5, 49.1] | 5500 [4956, 6059] | 1.1 [0.8, 1.5] | 0.4 [0.2, 0.7] |
| test_train_modes | trained - noise | -0.3 [-0.7, 0.2] | 1.1 [0.5, 1.7] | -25.2 [-28.1, -22.3] | -6675 [-8426, -4818] | 0.6 [0.0, 1.1] | 1.2 [0.7, 1.7] |
| test_train_modes | trained - untrained | -0.4 [-0.8, 0.0] | 0.9 [0.4, 1.5] | -28.1 [-30.7, -25.1] | -1533 [-2044, -1080] | 0.7 [0.2, 1.3] | 1.1 [0.6, 1.6] |

Seed spread (honest, %): trained [1.1, 0.7, 0.5]; noise [1.1, 1.3, 0.7]. compl_long bar = 2658.0 chars; 500 questions.

## fragility-cpt-cpt-qwen38b

fragility/cpt / cpt / Qwen/Qwen3-8B; 3/3 seeds registered.

KL(base || trained): seed0 0.1276 ± 0.0020, seed1 0.1283 ± 0.0020, seed2 0.1282 ± 0.0020; median 0.1282.  
Noise (frob, target 0.1282, shapes from cpt-qwen8b-r32-seed2): noise0 (seed 0, scale 18.4) 0.1355, noise1 (seed 1, scale 18.4) 0.1313, noise2 (seed 2, scale 18.4) 0.1338.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| test_baseline | trained (3 adapters, n=1500) | n/a | n/a | 35.1 [31.1, 38.8] | 14869 [13582, 16060] | 0.0 [0.0, 0.0] | n/a |
| test_baseline | noise (3 adapters, n=1500) | n/a | n/a | 36.7 [31.5, 42.0] | 19775 [16751, 24463] | 0.0 [0.0, 0.0] | n/a |
| test_baseline | untrained (3 adapters, n=1500) | n/a | n/a | 33.4 [29.3, 37.4] | 20433 [19136, 21657] | 0.0 [0.0, 0.0] | n/a |
| test_baseline | trained - noise | n/a | n/a | -1.6 [-6.4, 3.3] | -4906 [-9378, -2120] | 0.0 [0.0, 0.0] | n/a |
| test_baseline | trained - untrained | n/a | n/a | 1.7 [-2.3, 5.2] | -5564 [-6544, -4661] | 0.0 [0.0, 0.0] | n/a |

Seed spread (honest, %): trained [nan, nan, nan]; noise [nan, nan, nan]. compl_long bar = 17120.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| test_heldout | trained (3 adapters, n=4500) | 0.3 [0.1, 0.6] | 0.3 [0.1, 0.6] | 38.0 [34.6, 41.3] | 9503 [8679, 10312] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] |
| test_heldout | noise (3 adapters, n=4500) | 0.2 [0.0, 0.4] | 0.2 [0.0, 0.4] | 41.4 [37.4, 44.8] | 13843 [11261, 17916] | 1.0 [0.1, 2.1] | 0.0 [0.0, 0.0] |
| test_heldout | untrained (3 adapters, n=4500) | 0.1 [0.0, 0.3] | 0.1 [0.0, 0.3] | 41.7 [37.9, 45.3] | 14242 [13207, 15236] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| test_heldout | trained - noise | 0.1 [-0.2, 0.4] | 0.1 [-0.2, 0.4] | -3.3 [-5.6, -0.9] | -4340 [-8420, -1977] | -0.9 [-2.1, -0.1] | 0.0 [0.0, 0.0] |
| test_heldout | trained - untrained | 0.2 [-0.1, 0.5] | 0.2 [-0.1, 0.5] | -3.7 [-6.0, -1.3] | -4738 [-5365, -4132] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.5, 0.1, 0.3]; noise [0.1, 0.1, 0.3]. compl_long bar = 10241.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| test_train_modes | trained (3 adapters, n=13392) | 0.4 [0.2, 0.6] | 0.8 [0.6, 1.0] | 36.3 [33.2, 39.3] | 9513 [8793, 10218] | 10.9 [10.5, 11.3] | 0.9 [0.5, 1.4] |
| test_train_modes | noise (3 adapters, n=13392) | 0.1 [0.0, 0.3] | 0.3 [0.0, 0.7] | 38.4 [34.7, 41.7] | 13263 [10959, 16994] | 10.9 [10.3, 11.4] | 0.3 [0.0, 0.7] |
| test_train_modes | untrained (3 adapters, n=13392) | 0.6 [0.4, 0.8] | 1.6 [1.2, 2.0] | 39.4 [35.8, 42.6] | 13884 [12928, 14804] | 11.2 [11.0, 11.3] | 2.2 [1.7, 2.9] |
| test_train_modes | trained - noise | 0.3 [0.1, 0.4] | 0.5 [0.1, 0.8] | -2.1 [-3.8, -0.3] | -3750 [-7441, -1610] | -0.0 [-0.7, 0.7] | 0.6 [0.0, 1.2] |
| test_train_modes | trained - untrained | -0.2 [-0.4, 0.1] | -0.8 [-1.2, -0.4] | -3.1 [-4.9, -1.0] | -4371 [-4833, -3873] | -0.3 [-0.7, 0.1] | -1.3 [-2.1, -0.6] |

Seed spread (honest, %): trained [0.3, 0.4, 0.4]; noise [0.2, 0.0, 0.2]. compl_long bar = 9471.0 chars; 500 questions.

## fragility-cpt-cpt-step50-gptoss20b

fragility/cpt / cpt / openai/gpt-oss-20b; 3/3 seeds registered.

KL(base || trained): seed0 0.0701 ± 0.0016, seed1 0.0728 ± 0.0016, seed2 0.0727 ± 0.0016; median 0.0727.  
Noise (frob, target 0.0727, shapes from cpt-gptoss20b-r32-seed2-step50): noise0 (seed 0, scale 8.41) 0.0728, noise1 (seed 1, scale 7.18) 0.0711, noise2 (seed 2, scale 8.91) 0.0726.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| test_baseline | trained (3 adapters, n=1500) | n/a | n/a | 38.2 [33.9, 42.9] | 6011 [5236, 6863] | 0.7 [0.2, 1.4] | n/a |
| test_baseline | noise (3 adapters, n=1500) | n/a | n/a | 43.7 [39.0, 47.8] | 13727 [10971, 16084] | 0.6 [0.1, 1.3] | n/a |
| test_baseline | untrained (3 adapters, n=1500) | n/a | n/a | 49.3 [45.0, 53.4] | 7112 [6326, 7933] | 0.5 [0.0, 1.1] | n/a |
| test_baseline | trained - noise | n/a | n/a | -5.5 [-9.7, -1.0] | -7716 [-10002, -4924] | 0.1 [-0.7, 0.9] | n/a |
| test_baseline | trained - untrained | n/a | n/a | -11.1 [-15.3, -7.1] | -1101 [-1897, -293] | 0.3 [-0.5, 1.0] | n/a |

Seed spread (honest, %): trained [nan, nan, nan]; noise [nan, nan, nan]. compl_long bar = 3636.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| test_heldout | trained (3 adapters, n=4500) | 2.4 [1.7, 3.2] | 3.1 [2.2, 4.0] | 21.5 [18.6, 24.4] | 4896 [4453, 5404] | 1.2 [0.7, 1.7] | 0.3 [0.0, 0.7] |
| test_heldout | noise (3 adapters, n=4500) | 0.8 [0.4, 1.2] | 1.0 [0.5, 1.4] | 43.4 [39.3, 47.2] | 11645 [9904, 13468] | 1.2 [0.6, 1.9] | 0.0 [0.0, 0.0] |
| test_heldout | untrained (3 adapters, n=4500) | 1.1 [0.6, 1.6] | 1.2 [0.8, 1.8] | 45.6 [41.6, 49.4] | 5591 [5022, 6143] | 1.0 [0.5, 1.7] | 0.1 [0.0, 0.3] |
| test_heldout | trained - noise | 1.6 [0.7, 2.5] | 2.1 [1.1, 3.1] | -21.9 [-25.7, -18.1] | -6749 [-8559, -4989] | -0.1 [-0.9, 0.7] | 0.3 [0.0, 0.7] |
| test_heldout | trained - untrained | 1.4 [0.4, 2.2] | 1.8 [0.8, 2.8] | -24.1 [-27.6, -20.5] | -695 [-1317, -50] | 0.1 [-0.6, 0.8] | 0.2 [-0.2, 0.7] |

Seed spread (honest, %): trained [2.9, 1.8, 2.6]; noise [1.1, 0.9, 0.5]. compl_long bar = 2792.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| test_train_modes | trained (3 adapters, n=13392) | 0.8 [0.6, 1.1] | 1.7 [1.5, 2.0] | 22.9 [20.8, 25.0] | 4722 [4459, 4979] | 2.2 [1.8, 2.5] | 0.9 [0.6, 1.5] |
| test_train_modes | noise (3 adapters, n=13392) | 0.9 [0.5, 1.2] | 0.9 [0.6, 1.3] | 42.7 [38.7, 46.3] | 11718 [9880, 13430] | 1.5 [1.2, 1.9] | 0.3 [0.1, 0.5] |
| test_train_modes | untrained (3 adapters, n=13392) | 1.2 [0.9, 1.5] | 1.3 [1.0, 1.6] | 45.4 [41.5, 49.1] | 5500 [4956, 6059] | 1.1 [0.8, 1.5] | 0.4 [0.2, 0.7] |
| test_train_modes | trained - noise | -0.0 [-0.4, 0.4] | 0.8 [0.4, 1.2] | -19.7 [-22.5, -16.8] | -6996 [-8654, -5219] | 0.7 [0.2, 1.1] | 0.7 [0.2, 1.2] |
| test_train_modes | trained - untrained | -0.4 [-0.7, 0.0] | 0.5 [0.1, 0.8] | -22.5 [-25.0, -19.8] | -778 [-1216, -357] | 1.0 [0.5, 1.5] | 0.5 [0.0, 1.1] |

Seed spread (honest, %): trained [1.0, 0.8, 0.7]; noise [1.1, 1.0, 0.5]. compl_long bar = 2658.0 chars; 500 questions.

## fragility-cpt-cpt-step50-qwen38b

fragility/cpt / cpt / Qwen/Qwen3-8B; 3/3 seeds registered.

KL(base || trained): seed0 0.0862 ± 0.0015, seed1 0.0855 ± 0.0015, seed2 0.0850 ± 0.0015; median 0.0855.  
Noise (frob, target 0.0855, shapes from cpt-qwen8b-r32-seed1-step50): noise0 (seed 0, scale 27.1) 0.0842, noise1 (seed 1, scale 27.1) 0.0815, noise2 (seed 2, scale 27.1) 0.0829.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| test_baseline | trained (3 adapters, n=1500) | n/a | n/a | 34.5 [30.7, 38.3] | 18498 [17261, 19641] | 0.0 [0.0, 0.0] | n/a |
| test_baseline | noise (3 adapters, n=1500) | n/a | n/a | 35.8 [30.8, 40.5] | 19906 [17317, 23419] | 0.0 [0.0, 0.0] | n/a |
| test_baseline | untrained (3 adapters, n=1500) | n/a | n/a | 33.4 [29.3, 37.4] | 20433 [19136, 21657] | 0.0 [0.0, 0.0] | n/a |
| test_baseline | trained - noise | n/a | n/a | -1.3 [-5.7, 3.0] | -1408 [-4766, 776] | 0.0 [0.0, 0.0] | n/a |
| test_baseline | trained - untrained | n/a | n/a | 1.1 [-2.4, 4.5] | -1935 [-2624, -1263] | 0.0 [0.0, 0.0] | n/a |

Seed spread (honest, %): trained [nan, nan, nan]; noise [nan, nan, nan]. compl_long bar = 17120.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| test_heldout | trained (3 adapters, n=4500) | 0.1 [0.0, 0.2] | 0.1 [0.0, 0.2] | 40.1 [36.4, 43.8] | 11817 [10930, 12626] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] |
| test_heldout | noise (3 adapters, n=4500) | 0.2 [0.0, 0.4] | 0.2 [0.0, 0.4] | 40.6 [36.7, 44.2] | 14014 [11895, 17186] | 0.3 [0.1, 0.6] | 0.0 [0.0, 0.0] |
| test_heldout | untrained (3 adapters, n=4500) | 0.1 [0.0, 0.3] | 0.1 [0.0, 0.3] | 41.7 [37.9, 45.3] | 14242 [13207, 15236] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| test_heldout | trained - noise | -0.1 [-0.3, 0.1] | -0.1 [-0.3, 0.1] | -0.5 [-2.4, 1.4] | -2196 [-5272, -390] | -0.2 [-0.5, -0.0] | 0.0 [0.0, 0.0] |
| test_heldout | trained - untrained | -0.0 [-0.2, 0.2] | -0.0 [-0.2, 0.2] | -1.6 [-3.4, 0.3] | -2425 [-2878, -1959] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.0, 0.1, 0.1]; noise [0.3, 0.1, 0.1]. compl_long bar = 10241.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| test_train_modes | trained (3 adapters, n=13392) | 0.4 [0.3, 0.6] | 1.0 [0.7, 1.3] | 38.2 [34.9, 41.1] | 11599 [10771, 12367] | 11.1 [10.9, 11.3] | 1.2 [0.7, 1.7] |
| test_train_modes | noise (3 adapters, n=13392) | 0.2 [0.1, 0.4] | 0.5 [0.1, 1.0] | 39.6 [36.0, 42.7] | 13443 [11370, 16550] | 10.8 [10.4, 11.2] | 0.6 [0.0, 1.4] |
| test_train_modes | untrained (3 adapters, n=13392) | 0.6 [0.4, 0.8] | 1.6 [1.2, 2.0] | 39.4 [35.8, 42.6] | 13884 [12928, 14804] | 11.2 [11.0, 11.3] | 2.2 [1.7, 2.9] |
| test_train_modes | trained - noise | 0.2 [0.0, 0.4] | 0.5 [-0.0, 1.0] | -1.4 [-2.8, 0.1] | -1844 [-4956, -57] | 0.3 [-0.1, 0.7] | 0.6 [-0.4, 1.4] |
| test_train_modes | trained - untrained | -0.1 [-0.4, 0.1] | -0.6 [-1.0, -0.2] | -1.2 [-2.5, 0.4] | -2285 [-2631, -1941] | -0.1 [-0.3, 0.2] | -1.0 [-1.8, -0.3] |

Seed spread (honest, %): trained [0.4, 0.5, 0.4]; noise [0.3, 0.1, 0.3]. compl_long bar = 9471.0 chars; 500 questions.

## fragility-mcq_noreason-gptoss20b

fragility / mcq_noreason / openai/gpt-oss-20b; 3/3 seeds registered.

KL(base || trained): seed0 0.0820 ± 0.0018, seed1 0.0491 ± 0.0012, seed2 0.0609 ± 0.0014; median 0.0609.  
Noise (frob, target 0.0609, shapes from fragility-mcq_noreason-gptoss20b-seed2): noise0 (seed 0, scale 11.1) 0.0551, noise1 (seed 1, scale 9.69) 0.0560, noise2 (seed 2, scale 12.8) 0.0633.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_baseline | trained (3 adapters, n=1500) | n/a | n/a | 39.9 [34.5, 45.2] | 3749 [3227, 4445] | 1.6 [0.7, 2.7] | n/a |
| final_baseline | noise (3 adapters, n=1500) | n/a | n/a | 44.4 [40.0, 48.8] | 13233 [10981, 15226] | 0.5 [0.1, 1.1] | n/a |
| final_baseline | untrained (1 adapters, n=500) | n/a | n/a | 48.6 [44.0, 52.8] | 6922 [6053, 7744] | 1.0 [0.2, 2.0] | n/a |
| final_baseline | trained - noise | n/a | n/a | -4.5 [-9.7, 0.7] | -9484 [-11426, -7323] | 1.1 [0.2, 2.1] | n/a |
| final_baseline | trained - untrained | n/a | n/a | -8.7 [-14.1, -3.9] | -3173 [-3948, -2354] | 0.6 [-0.6, 1.8] | n/a |

Seed spread (honest, %): trained [nan, nan, nan]; noise [nan, nan, nan]. compl_long bar = 3194.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_heldout | trained (3 adapters, n=4500) | 4.0 [2.8, 5.3] | 5.3 [4.3, 6.4] | 33.2 [28.0, 38.1] | 2270 [2037, 2534] | 1.6 [1.0, 2.3] | 0.0 [0.0, 0.0] |
| final_heldout | noise (3 adapters, n=4500) | 0.7 [0.3, 1.2] | 0.9 [0.4, 1.7] | 43.5 [39.3, 47.3] | 10996 [9359, 12841] | 1.2 [0.5, 2.0] | 0.0 [0.0, 0.0] |
| final_heldout | untrained (1 adapters, n=1500) | 1.3 [0.7, 1.9] | 1.5 [0.9, 2.1] | 45.3 [41.1, 49.0] | 5744 [5174, 6378] | 1.1 [0.5, 1.7] | 0.0 [0.0, 0.0] |
| final_heldout | trained - noise | 3.2 [2.0, 4.7] | 4.4 [3.2, 5.6] | -10.2 [-15.5, -5.8] | -8726 [-10527, -7163] | 0.4 [-0.5, 1.3] | 0.0 [0.0, 0.0] |
| final_heldout | trained - untrained | 2.7 [1.4, 4.1] | 3.8 [2.7, 5.1] | -12.0 [-17.0, -7.9] | -3474 [-4011, -3001] | 0.5 [-0.2, 1.3] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [5.2, 3.7, 3.1]; noise [1.1, 0.5, 0.5]. compl_long bar = 2734.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_modes | trained (3 adapters, n=13392) | 4.9 [4.0, 5.8] | 6.3 [5.5, 6.9] | 32.7 [27.0, 38.0] | 2515 [2244, 2872] | 2.7 [1.6, 3.9] | 2.2 [1.5, 3.0] |
| final_modes | noise (3 adapters, n=13392) | 0.9 [0.5, 1.3] | 1.0 [0.6, 1.4] | 42.7 [38.7, 46.4] | 10829 [9336, 12444] | 1.3 [0.9, 1.6] | 0.4 [0.2, 0.6] |
| final_modes | untrained (1 adapters, n=4464) | 1.5 [1.1, 1.9] | 1.6 [1.2, 2.1] | 46.1 [42.1, 49.7] | 5499 [4970, 6054] | 1.0 [0.7, 1.4] | 0.5 [0.2, 0.8] |
| final_modes | trained - noise | 4.0 [3.1, 4.9] | 5.3 [4.5, 6.0] | -10.0 [-15.5, -5.4] | -8314 [-9859, -6905] | 1.5 [0.4, 2.6] | 1.9 [1.1, 2.6] |
| final_modes | trained - untrained | 3.4 [2.5, 4.3] | 4.6 [3.8, 5.3] | -13.4 [-18.9, -8.9] | -2984 [-3450, -2550] | 1.7 [0.6, 2.9] | 1.7 [1.0, 2.5] |

Seed spread (honest, %): trained [5.0, 5.7, 4.1]; noise [1.1, 1.1, 0.5]. compl_long bar = 2612.0 chars; 500 questions.

## fragility-mcq_noreason-qwen3627b

fragility / mcq_noreason / Qwen/Qwen3.6-27B; 5/5 seeds registered.

KL(base || trained): seed0 0.0052 ± 0.0003, seed1 0.0062 ± 0.0004, seed2 0.0065 ± 0.0004, seed3 0.0063 ± 0.0004, seed4 0.0028 ± 0.0002; median 0.0062.  
Noise (frob, target 0.0062, shapes from fragility-mcq_noreason-qwen36_27b-seed1): noise0 (seed 0, scale 14.2) 0.0061, noise1 (seed 1, scale 14.2) 0.0065, noise2 (seed 2, scale 14.2) 0.0061.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_baseline | trained (5 adapters, n=2500) | n/a | n/a | 42.0 [37.9, 46.1] | 23181 [21271, 25110] | 0.0 [0.0, 0.0] | n/a |
| final_baseline | noise (3 adapters, n=1500) | n/a | n/a | 39.3 [35.0, 43.9] | 27033 [25444, 28602] | 0.0 [0.0, 0.0] | n/a |
| final_baseline | untrained (1 adapters, n=500) | n/a | n/a | 39.6 [35.4, 44.0] | 27073 [25495, 28629] | 0.0 [0.0, 0.0] | n/a |
| final_baseline | trained - noise | n/a | n/a | 2.7 [-0.3, 6.0] | -3852 [-5513, -2239] | 0.0 [0.0, 0.0] | n/a |
| final_baseline | trained - untrained | n/a | n/a | 2.4 [-0.6, 5.6] | -3892 [-5534, -2204] | 0.0 [0.0, 0.0] | n/a |

Seed spread (honest, %): trained [nan, nan, nan, nan, nan]; noise [nan, nan, nan]. compl_long bar = 25212.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_alternating_case | trained (5 adapters, n=2500) | 1.8 [1.0, 2.8] | 2.2 [1.2, 3.3] | 49.1 [44.9, 53.1] | 10974 [9471, 12719] | 13.2 [10.6, 16.3] | 0.0 [0.0, 0.0] |
| final_direct_alternating_case | noise (3 adapters, n=1500) | 1.4 [0.6, 2.3] | 1.5 [0.7, 2.5] | 45.1 [41.3, 48.9] | 16141 [13854, 18787] | 13.1 [9.9, 16.9] | 0.0 [0.0, 0.0] |
| final_direct_alternating_case | untrained (1 adapters, n=500) | 1.2 [0.4, 2.2] | 1.4 [0.6, 2.6] | 48.2 [43.6, 52.4] | 13812 [12300, 15466] | 12.8 [9.8, 15.6] | 0.0 [0.0, 0.0] |
| final_direct_alternating_case | trained - noise | 0.4 [-0.8, 1.7] | 0.6 [-0.7, 1.9] | 4.0 [0.8, 7.0] | -5167 [-7895, -2636] | 0.1 [-4.4, 4.5] | 0.0 [0.0, 0.0] |
| final_direct_alternating_case | trained - untrained | 0.6 [-0.6, 1.8] | 0.8 [-0.7, 2.2] | 0.9 [-2.7, 4.4] | -2838 [-4470, -1180] | 0.4 [-3.3, 4.6] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.8, 2.6, 2.0, 1.6, 2.2]; noise [1.0, 1.2, 2.0]. compl_long bar = 2862.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_end_of_sentence | trained (5 adapters, n=2500) | 2.6 [1.2, 4.1] | 2.6 [1.2, 4.2] | 50.3 [46.3, 54.5] | 13725 [10789, 16312] | 8.7 [4.1, 14.2] | 0.0 [0.0, 0.0] |
| final_direct_end_of_sentence | noise (3 adapters, n=1500) | 2.0 [0.8, 3.5] | 2.0 [0.8, 3.5] | 46.1 [41.9, 50.1] | 17718 [15643, 19495] | 10.0 [8.0, 12.2] | 0.0 [0.0, 0.0] |
| final_direct_end_of_sentence | untrained (1 adapters, n=500) | 0.6 [0.0, 1.2] | 0.8 [0.2, 1.6] | 44.8 [40.2, 49.4] | 16746 [15148, 18381] | 11.6 [8.8, 14.6] | 0.0 [0.0, 0.0] |
| final_direct_end_of_sentence | trained - noise | 0.6 [-1.3, 2.6] | 0.6 [-1.3, 2.6] | 4.3 [1.6, 6.9] | -3993 [-7061, -1105] | -1.3 [-6.3, 4.3] | 0.0 [0.0, 0.0] |
| final_direct_end_of_sentence | trained - untrained | 2.0 [0.4, 3.6] | 1.8 [0.2, 3.6] | 5.5 [2.7, 8.8] | -3021 [-6122, -458] | -2.9 [-8.2, 3.1] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [2.0, 4.6, 1.0, 1.6, 3.8]; noise [3.2, 1.0, 1.8]. compl_long bar = 7549.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_lowercase_thinking | trained (5 adapters, n=2500) | 7.0 [3.3, 12.1] | 7.4 [3.6, 12.6] | 51.5 [47.2, 55.6] | 13856 [11967, 15970] | 6.3 [4.6, 8.3] | 0.9 [0.0, 2.4] |
| final_direct_lowercase_thinking | noise (3 adapters, n=1500) | 2.9 [1.7, 4.2] | 3.1 [1.9, 4.4] | 46.8 [42.3, 50.9] | 19729 [17595, 21850] | 5.0 [3.2, 6.9] | 0.3 [0.0, 0.8] |
| final_direct_lowercase_thinking | untrained (1 adapters, n=500) | 3.2 [1.8, 4.8] | 3.4 [2.0, 5.0] | 47.4 [42.8, 51.6] | 19904 [18226, 21632] | 6.2 [4.2, 8.2] | 0.4 [0.0, 1.3] |
| final_direct_lowercase_thinking | trained - noise | 4.2 [0.2, 9.3] | 4.4 [0.3, 9.5] | 4.7 [1.9, 8.1] | -5872 [-8212, -3369] | 1.3 [-1.3, 4.1] | 0.6 [-0.5, 2.2] |
| final_direct_lowercase_thinking | trained - untrained | 3.8 [-0.4, 9.1] | 4.0 [-0.3, 9.4] | 4.1 [0.6, 7.4] | -6048 [-8086, -4022] | 0.1 [-2.6, 2.9] | 0.5 [-0.8, 2.1] |

Seed spread (honest, %): trained [2.8, 7.2, 16.8, 3.0, 5.4]; noise [3.0, 2.6, 3.0]. compl_long bar = 12099.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_uppercase_thinking | trained (5 adapters, n=2500) | 2.4 [1.3, 4.0] | 2.5 [1.3, 4.2] | 49.7 [45.4, 53.6] | 17879 [15083, 20368] | 3.3 [1.8, 5.1] | 0.2 [0.0, 0.7] |
| final_direct_uppercase_thinking | noise (3 adapters, n=1500) | 3.5 [2.2, 5.0] | 3.5 [2.2, 5.2] | 48.8 [44.5, 52.9] | 16957 [15256, 18572] | 6.2 [4.0, 8.8] | 0.0 [0.0, 0.0] |
| final_direct_uppercase_thinking | untrained (1 adapters, n=500) | 2.0 [1.0, 3.2] | 2.2 [1.0, 3.4] | 44.4 [40.0, 48.6] | 21351 [19756, 23043] | 4.6 [2.8, 6.4] | 0.0 [0.0, 0.0] |
| final_direct_uppercase_thinking | trained - noise | -1.1 [-3.0, 0.8] | -1.0 [-3.0, 1.0] | 0.9 [-2.2, 3.9] | 922 [-1953, 3621] | -2.9 [-5.9, -0.2] | 0.2 [0.0, 0.7] |
| final_direct_uppercase_thinking | trained - untrained | 0.4 [-1.2, 2.1] | 0.3 [-1.4, 2.2] | 5.3 [2.2, 8.6] | -3472 [-6236, -926] | -1.3 [-3.6, 0.9] | 0.2 [0.0, 0.7] |

Seed spread (honest, %): trained [1.2, 4.4, 1.6, 2.6, 2.2]; noise [4.4, 3.0, 3.0]. compl_long bar = 14380.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_heldout | trained (5 adapters, n=7500) | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.1] | 51.5 [47.4, 55.3] | 19067 [18122, 19987] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] |
| final_heldout | noise (3 adapters, n=4500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 48.5 [44.1, 52.6] | 22409 [21361, 23469] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] |
| final_heldout | untrained (1 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 49.3 [45.5, 52.9] | 22180 [21306, 23125] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_heldout | trained - noise | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.1] | 2.9 [0.4, 5.6] | -3343 [-4347, -2360] | -0.0 [-0.1, 0.1] | 0.0 [0.0, 0.0] |
| final_heldout | trained - untrained | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.1] | 2.2 [0.0, 4.4] | -3113 [-3914, -2254] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.0, 0.0, 0.1, 0.0, 0.0]; noise [0.0, 0.0, 0.0]. compl_long bar = 17061.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_modes | trained (5 adapters, n=22320) | 1.1 [0.7, 1.5] | 1.2 [0.8, 1.6] | 48.0 [44.3, 51.7] | 19681 [18652, 20693] | 0.1 [0.0, 0.1] | 0.0 [0.0, 0.1] |
| final_modes | noise (3 adapters, n=13392) | 0.3 [0.1, 0.5] | 0.3 [0.2, 0.6] | 45.3 [41.6, 49.0] | 24035 [23112, 24982] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.1] |
| final_modes | untrained (1 adapters, n=4464) | 0.3 [0.2, 0.5] | 0.4 [0.2, 0.6] | 45.9 [42.1, 49.5] | 23480 [22555, 24474] | 0.0 [0.0, 0.1] | 0.1 [0.0, 0.2] |
| final_modes | trained - noise | 0.8 [0.4, 1.2] | 0.9 [0.4, 1.3] | 2.6 [1.2, 4.1] | -4354 [-5090, -3698] | 0.1 [-0.0, 0.1] | -0.0 [-0.1, 0.0] |
| final_modes | trained - untrained | 0.8 [0.4, 1.2] | 0.8 [0.4, 1.2] | 2.1 [0.6, 3.5] | -3799 [-4564, -3111] | 0.0 [-0.1, 0.1] | -0.1 [-0.2, 0.0] |

Seed spread (honest, %): trained [1.3, 1.2, 1.1, 1.6, 0.5]; noise [0.2, 0.5, 0.2]. compl_long bar = 17794.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_alternating_case | trained (5 adapters, n=2500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 53.8 [49.7, 57.9] | 13882 [12123, 15852] | 0.2 [0.0, 0.5] | 0.0 [0.0, 0.0] |
| final_persona_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 53.5 [48.8, 57.9] | 14744 [13516, 15995] | 0.1 [0.0, 0.4] | 0.0 [0.0, 0.0] |
| final_persona_alternating_case | untrained (1 adapters, n=500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 52.4 [48.0, 56.8] | 14726 [13705, 15911] | 0.6 [0.0, 1.4] | 0.0 [0.0, 0.0] |
| final_persona_alternating_case | trained - noise | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.4 [-2.6, 3.6] | -863 [-2711, 1062] | 0.1 [-0.3, 0.5] | 0.0 [0.0, 0.0] |
| final_persona_alternating_case | trained - untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 1.4 [-1.3, 4.2] | -844 [-2532, 1016] | -0.4 [-1.2, 0.2] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.0, 0.0, 0.0, 0.0, 0.0]; noise [0.0, 0.0, 0.0]. compl_long bar = 9599.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_end_of_sentence | trained (5 adapters, n=2500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 53.2 [49.3, 57.1] | 13239 [11279, 15289] | 0.0 [0.0, 0.2] | 0.0 [0.0, 0.0] |
| final_persona_end_of_sentence | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 53.8 [49.9, 58.0] | 14371 [13142, 15667] | 0.1 [0.0, 0.3] | 0.0 [0.0, 0.0] |
| final_persona_end_of_sentence | untrained (1 adapters, n=500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 52.8 [48.2, 57.0] | 14300 [13147, 15404] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_persona_end_of_sentence | trained - noise | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | -0.6 [-3.3, 1.9] | -1133 [-3287, 978] | -0.0 [-0.3, 0.2] | 0.0 [0.0, 0.0] |
| final_persona_end_of_sentence | trained - untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.4 [-2.6, 3.2] | -1061 [-3082, 967] | 0.0 [0.0, 0.2] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.0, 0.0, 0.0, 0.0, 0.0]; noise [0.0, 0.0, 0.0]. compl_long bar = 9519.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_lowercase_thinking | trained (5 adapters, n=2500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 53.5 [49.4, 57.6] | 14334 [12485, 16205] | 0.1 [0.0, 0.4] | 0.0 [0.0, 0.0] |
| final_persona_lowercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 52.7 [48.3, 56.7] | 15979 [14754, 17272] | 0.3 [0.0, 0.8] | 0.0 [0.0, 0.0] |
| final_persona_lowercase_thinking | untrained (1 adapters, n=500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 52.4 [48.0, 56.6] | 15064 [13857, 16223] | 0.2 [0.0, 0.6] | 0.0 [0.0, 0.0] |
| final_persona_lowercase_thinking | trained - noise | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.9 [-1.9, 3.7] | -1646 [-3563, 140] | -0.2 [-0.7, 0.2] | 0.0 [0.0, 0.0] |
| final_persona_lowercase_thinking | trained - untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 1.1 [-1.9, 4.2] | -730 [-2619, 1221] | -0.1 [-0.6, 0.3] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.0, 0.0, 0.0, 0.0, 0.0]; noise [0.0, 0.0, 0.0]. compl_long bar = 9800.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_uppercase_thinking | trained (5 adapters, n=2500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 53.8 [49.4, 57.9] | 14351 [12363, 16275] | 0.4 [0.1, 1.0] | 0.0 [0.0, 0.0] |
| final_persona_uppercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.5 [47.1, 55.9] | 15924 [14661, 17073] | 0.3 [0.0, 0.9] | 0.0 [0.0, 0.0] |
| final_persona_uppercase_thinking | untrained (1 adapters, n=500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 54.4 [50.2, 58.4] | 15037 [13673, 16164] | 0.6 [0.0, 1.4] | 0.0 [0.0, 0.0] |
| final_persona_uppercase_thinking | trained - noise | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 2.3 [-1.2, 5.5] | -1572 [-3604, 448] | 0.1 [-0.6, 0.8] | 0.0 [0.0, 0.0] |
| final_persona_uppercase_thinking | trained - untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | -0.6 [-4.4, 2.6] | -686 [-2656, 1287] | -0.2 [-0.9, 0.6] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.0, 0.0, 0.0, 0.0, 0.0]; noise [0.0, 0.0, 0.0]. compl_long bar = 10201.0 chars; 500 questions.

## fragility-mcq_noreason-qwen38b

fragility / mcq_noreason / Qwen/Qwen3-8B; 3/3 seeds registered.

KL(base || trained): seed0 0.0246 ± 0.0006, seed1 0.0394 ± 0.0009, seed2 0.0281 ± 0.0007; median 0.0281.  
Noise (frob, target 0.0281, shapes from fragility-mcq_noreason-qwen8b-seed2): noise0 (seed 0, scale 23.9) 0.0275, noise1 (seed 1, scale 23.9) 0.0265, noise2 (seed 2, scale 23.9) 0.0267.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_baseline | trained (3 adapters, n=1500) | n/a | n/a | 29.8 [24.8, 34.6] | 8570 [721, 13270] | 31.3 [0.0, 93.4] | n/a |
| final_baseline | noise (3 adapters, n=1500) | n/a | n/a | 33.9 [30.1, 38.0] | 20219 [18388, 22624] | 0.0 [0.0, 0.0] | n/a |
| final_baseline | untrained (1 adapters, n=500) | n/a | n/a | 31.4 [27.4, 35.2] | 20748 [19431, 21994] | 0.0 [0.0, 0.0] | n/a |
| final_baseline | trained - noise | n/a | n/a | -4.1 [-9.8, 0.9] | -11649 [-19171, -6461] | 31.3 [0.0, 93.4] | n/a |
| final_baseline | trained - untrained | n/a | n/a | -1.6 [-7.2, 4.0] | -12178 [-19918, -7583] | 31.3 [0.0, 93.4] | n/a |

Seed spread (honest, %): trained [nan, nan, nan]; noise [nan, nan, nan]. compl_long bar = 17232.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_heldout | trained (3 adapters, n=4500) | 1.4 [0.5, 2.7] | 1.9 [0.7, 3.5] | 30.9 [26.8, 35.0] | 5105 [1451, 7375] | 19.7 [1.1, 55.5] | 0.3 [0.0, 3.5] |
| final_heldout | noise (3 adapters, n=4500) | 0.1 [0.0, 0.3] | 0.1 [0.0, 0.3] | 42.4 [38.4, 46.2] | 14254 [12650, 16350] | 0.1 [0.0, 0.3] | 0.0 [0.0, 0.0] |
| final_heldout | untrained (1 adapters, n=1500) | 0.1 [0.0, 0.3] | 0.1 [0.0, 0.3] | 42.5 [38.4, 46.3] | 13940 [12903, 14910] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_heldout | trained - noise | 1.3 [0.3, 2.5] | 1.8 [0.6, 3.4] | -11.6 [-15.7, -7.9] | -9149 [-12772, -6213] | 19.6 [1.0, 55.4] | 0.3 [0.0, 3.5] |
| final_heldout | trained - untrained | 1.3 [0.3, 2.5] | 1.8 [0.6, 3.4] | -11.6 [-15.8, -7.7] | -8836 [-12266, -6491] | 19.7 [1.1, 55.5] | 0.3 [0.0, 3.5] |

Seed spread (honest, %): trained [1.0, 2.7, 0.5]; noise [0.1, 0.1, 0.2]. compl_long bar = 9880.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_modes | trained (3 adapters, n=13392) | 2.6 [1.1, 4.0] | 10.7 [5.8, 14.6] | 27.3 [23.7, 30.8] | 6749 [4035, 8463] | 24.5 [9.8, 46.6] | 22.0 [9.5, 47.8] |
| final_modes | noise (3 adapters, n=13392) | 0.4 [0.2, 0.6] | 1.4 [0.6, 2.4] | 38.9 [35.2, 42.1] | 13761 [12256, 15592] | 11.1 [10.9, 11.2] | 2.2 [0.8, 4.0] |
| final_modes | untrained (1 adapters, n=4464) | 0.5 [0.3, 0.7] | 1.5 [1.2, 1.8] | 39.9 [36.2, 43.1] | 13983 [13001, 14887] | 11.1 [10.9, 11.2] | 2.2 [1.7, 2.8] |
| final_modes | trained - noise | 2.2 [0.8, 3.6] | 9.3 [4.8, 13.2] | -11.6 [-15.2, -8.2] | -7012 [-9729, -4660] | 13.4 [-1.4, 35.5] | 19.8 [7.9, 45.2] |
| final_modes | trained - untrained | 2.2 [0.7, 3.5] | 9.2 [4.3, 13.2] | -12.5 [-16.2, -9.1] | -7234 [-9793, -5453] | 13.4 [-1.3, 35.5] | 19.8 [7.3, 45.9] |

Seed spread (honest, %): trained [2.9, 3.9, 1.1]; noise [0.3, 0.3, 0.6]. compl_long bar = 9647.0 chars; 500 questions.

## fragility-mcq_reason-gptoss20b

fragility / mcq_reason / openai/gpt-oss-20b; 3/3 seeds registered.

KL(base || trained): seed0 0.0040 ± 0.0004, seed1 0.0023 ± 0.0004, seed2 0.0048 ± 0.0004; median 0.0040.  
Noise (frob, target 0.0040, shapes from fragility-mcq_reason-gptoss20b-seed0): noise0 (seed 0, scale 2.8) 0.0041, noise1 (seed 1, scale 2.43) 0.0039, noise2 (seed 2, scale 2.8) 0.0040.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_baseline | trained (3 adapters, n=1500) | n/a | n/a | 47.2 [42.9, 51.4] | 6495 [5475, 7687] | 0.5 [0.0, 1.2] | n/a |
| final_baseline | noise (3 adapters, n=1500) | n/a | n/a | 46.9 [42.3, 50.9] | 7991 [7038, 8934] | 0.6 [0.1, 1.3] | n/a |
| final_baseline | untrained (1 adapters, n=500) | n/a | n/a | 47.4 [43.0, 51.6] | 7336 [6514, 8197] | 0.2 [0.0, 0.6] | n/a |
| final_baseline | trained - noise | n/a | n/a | 0.3 [-2.4, 3.1] | -1496 [-2448, -378] | -0.1 [-0.8, 0.5] | n/a |
| final_baseline | trained - untrained | n/a | n/a | -0.2 [-3.1, 2.8] | -840 [-1758, 253] | 0.3 [0.0, 0.8] | n/a |

Seed spread (honest, %): trained [nan, nan, nan]; noise [nan, nan, nan]. compl_long bar = 3762.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_heldout | trained (3 adapters, n=4500) | 0.7 [0.4, 1.2] | 0.8 [0.4, 1.3] | 46.4 [42.4, 50.1] | 5445 [4680, 6307] | 0.7 [0.3, 1.3] | 0.0 [0.0, 0.0] |
| final_heldout | noise (3 adapters, n=4500) | 1.0 [0.6, 1.3] | 1.1 [0.8, 1.6] | 45.9 [41.7, 49.6] | 6310 [5647, 7042] | 1.1 [0.7, 1.6] | 0.1 [0.0, 0.4] |
| final_heldout | untrained (1 adapters, n=1500) | 1.2 [0.7, 1.7] | 1.3 [0.8, 1.9] | 45.6 [41.5, 49.3] | 5291 [4768, 5804] | 0.5 [0.2, 0.8] | 0.0 [0.0, 0.0] |
| final_heldout | trained - noise | -0.2 [-0.7, 0.3] | -0.3 [-0.8, 0.2] | 0.5 [-1.4, 2.1] | -865 [-1520, -162] | -0.4 [-0.9, 0.2] | -0.1 [-0.4, 0.0] |
| final_heldout | trained - untrained | -0.5 [-1.1, 0.1] | -0.5 [-1.1, 0.1] | 0.8 [-1.1, 2.7] | 153 [-406, 828] | 0.3 [-0.1, 0.8] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.5, 0.9, 0.7]; noise [0.9, 0.9, 1.1]. compl_long bar = 2708.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_modes | trained (3 adapters, n=13392) | 1.2 [0.9, 1.5] | 1.3 [1.0, 1.6] | 46.3 [42.5, 50.0] | 5368 [4618, 6178] | 0.8 [0.4, 1.2] | 0.4 [0.2, 0.7] |
| final_modes | noise (3 adapters, n=13392) | 1.2 [1.0, 1.5] | 1.3 [1.0, 1.6] | 45.7 [41.7, 49.4] | 6009 [5426, 6632] | 1.1 [0.8, 1.5] | 0.4 [0.2, 0.7] |
| final_modes | untrained (1 adapters, n=4464) | 1.2 [0.9, 1.5] | 1.3 [1.0, 1.6] | 45.7 [41.8, 49.3] | 5527 [5027, 6078] | 1.3 [0.9, 1.7] | 0.3 [0.1, 0.5] |
| final_modes | trained - noise | -0.0 [-0.4, 0.3] | -0.0 [-0.3, 0.3] | 0.6 [-0.4, 1.5] | -640 [-1168, 12] | -0.3 [-0.6, -0.0] | -0.0 [-0.3, 0.2] |
| final_modes | trained - untrained | 0.0 [-0.3, 0.3] | -0.0 [-0.4, 0.3] | 0.6 [-0.5, 1.6] | -159 [-681, 511] | -0.5 [-0.8, -0.1] | 0.1 [-0.1, 0.4] |

Seed spread (honest, %): trained [1.2, 1.1, 1.3]; noise [1.2, 1.3, 1.1]. compl_long bar = 2620.0 chars; 500 questions.

## fragility-mcq_reason-qwen38b

fragility / mcq_reason / Qwen/Qwen3-8B; 3/3 seeds registered.

KL(base || trained): seed0 0.0007 ± 0.0001, seed1 0.0006 ± 0.0001, seed2 0.0007 ± 0.0001; median 0.0007.  
Noise (frob, target 0.0007, shapes from fragility-mcq_reason-qwen8b-seed2): noise0 (seed 0, scale 4) 0.0006, noise1 (seed 1, scale 4.6) 0.0007, noise2 (seed 2, scale 4.63) 0.0006.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_baseline | trained (3 adapters, n=1500) | n/a | n/a | 32.5 [27.7, 36.7] | 20738 [19348, 22004] | 0.0 [0.0, 0.0] | n/a |
| final_baseline | noise (3 adapters, n=1500) | n/a | n/a | 32.6 [28.5, 36.7] | 20376 [19003, 21643] | 0.0 [0.0, 0.0] | n/a |
| final_baseline | untrained (1 adapters, n=500) | n/a | n/a | 34.2 [29.8, 38.4] | 20833 [19509, 22090] | 0.0 [0.0, 0.0] | n/a |
| final_baseline | trained - noise | n/a | n/a | -0.1 [-3.9, 3.5] | 363 [-335, 1070] | 0.0 [0.0, 0.0] | n/a |
| final_baseline | trained - untrained | n/a | n/a | -1.7 [-5.5, 2.0] | -94 [-821, 566] | 0.0 [0.0, 0.0] | n/a |

Seed spread (honest, %): trained [nan, nan, nan]; noise [nan, nan, nan]. compl_long bar = 17206.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_heldout | trained (3 adapters, n=4500) | 0.1 [0.0, 0.3] | 0.1 [0.0, 0.3] | 42.6 [38.6, 46.3] | 14276 [13211, 15285] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] |
| final_heldout | noise (3 adapters, n=4500) | 0.3 [0.0, 0.6] | 0.3 [0.0, 0.6] | 42.2 [37.9, 45.8] | 14271 [13218, 15300] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_heldout | untrained (1 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 43.3 [39.3, 46.9] | 14053 [12994, 15084] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_heldout | trained - noise | -0.2 [-0.5, 0.1] | -0.2 [-0.5, 0.1] | 0.4 [-1.3, 2.0] | 5 [-347, 349] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] |
| final_heldout | trained - untrained | 0.1 [0.0, 0.3] | 0.1 [0.0, 0.3] | -0.7 [-2.7, 1.1] | 223 [-142, 595] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.2, 0.1, 0.1]; noise [0.5, 0.1, 0.3]. compl_long bar = 9828.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_modes | trained (3 adapters, n=13392) | 0.5 [0.3, 0.7] | 1.3 [1.0, 1.7] | 39.4 [35.9, 42.8] | 13827 [12852, 14744] | 11.1 [11.0, 11.2] | 1.9 [1.4, 2.5] |
| final_modes | noise (3 adapters, n=13392) | 0.5 [0.3, 0.7] | 1.3 [1.0, 1.6] | 39.8 [36.2, 43.0] | 13757 [12805, 14711] | 11.1 [11.0, 11.3] | 1.8 [1.3, 2.4] |
| final_modes | untrained (1 adapters, n=4464) | 0.4 [0.2, 0.6] | 1.4 [1.1, 1.7] | 39.5 [36.0, 42.9] | 13936 [12989, 14865] | 11.1 [10.9, 11.2] | 2.1 [1.5, 2.7] |
| final_modes | trained - noise | 0.0 [-0.1, 0.2] | 0.0 [-0.3, 0.3] | -0.3 [-1.3, 0.7] | 69 [-270, 414] | -0.0 [-0.1, 0.1] | 0.1 [-0.5, 0.7] |
| final_modes | trained - untrained | 0.1 [-0.1, 0.4] | -0.0 [-0.4, 0.3] | -0.1 [-1.0, 0.9] | -109 [-414, 207] | 0.0 [-0.1, 0.2] | -0.1 [-0.7, 0.4] |

Seed spread (honest, %): trained [0.6, 0.4, 0.4]; noise [0.5, 0.4, 0.4]. compl_long bar = 9473.0 chars; 500 questions.

## fragility-style_sft-gptoss20b

fragility / style_sft / openai/gpt-oss-20b; 3/3 seeds registered.

KL(base || trained): seed0 0.0730 ± 0.0025, seed1 0.0787 ± 0.0027, seed2 0.0727 ± 0.0025; median 0.0730.  
Noise (frob, target 0.0730, shapes from fragility-style_sft-gptoss20b-seed0): noise0 (seed 0, scale 10.3) 0.0692, noise1 (seed 1, scale 8.98) 0.0715, noise2 (seed 2, scale 11) 0.0728.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_baseline | trained (3 adapters, n=1500) | n/a | n/a | 0.0 [0.0, 0.0] | 2017 [1463, 2538] | 30.5 [22.4, 40.6] | n/a |
| final_baseline | noise (3 adapters, n=1500) | n/a | n/a | 43.3 [38.9, 47.2] | 13572 [11839, 15399] | 0.7 [0.2, 1.5] | n/a |
| final_baseline | untrained (1 adapters, n=500) | n/a | n/a | 46.0 [41.6, 50.6] | 7373 [6494, 8289] | 1.2 [0.4, 2.2] | n/a |
| final_baseline | trained - noise | n/a | n/a | -43.3 [-47.2, -38.9] | -11554 [-13320, -9853] | 29.7 [21.8, 39.9] | n/a |
| final_baseline | trained - untrained | n/a | n/a | -46.0 [-50.6, -41.6] | -5356 [-6310, -4465] | 29.3 [21.4, 39.6] | n/a |

Seed spread (honest, %): trained [nan, nan, nan]; noise [nan, nan, nan]. compl_long bar = 3389.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_heldout | trained (3 adapters, n=4500) | 0.0 [0.0, 0.1] | 22.2 [20.3, 24.3] | 0.0 [0.0, 0.0] | 488 [383, 587] | 41.0 [30.8, 53.3] | 0.0 [0.0, 0.0] |
| final_heldout | noise (3 adapters, n=4500) | 0.5 [0.2, 0.8] | 0.7 [0.3, 1.1] | 43.3 [39.0, 47.3] | 11674 [9904, 13621] | 1.2 [0.6, 2.0] | 0.0 [0.0, 0.0] |
| final_heldout | untrained (1 adapters, n=1500) | 1.2 [0.7, 1.7] | 1.2 [0.7, 1.7] | 45.4 [41.2, 49.2] | 5832 [5243, 6421] | 0.7 [0.3, 1.3] | 0.0 [0.0, 0.0] |
| final_heldout | trained - noise | -0.5 [-0.8, -0.2] | 21.5 [19.6, 23.7] | -43.3 [-47.3, -39.0] | -11186 [-13112, -9445] | 39.8 [29.5, 51.9] | 0.0 [0.0, 0.0] |
| final_heldout | trained - untrained | -1.2 [-1.7, -0.6] | 21.0 [19.1, 23.2] | -45.4 [-49.2, -41.2] | -5344 [-5934, -4765] | 40.3 [30.1, 52.5] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.0, 0.0, 0.1]; noise [0.5, 0.6, 0.3]. compl_long bar = 3021.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_modes | trained (3 adapters, n=13392) | 0.0 [0.0, 0.1] | 18.7 [16.8, 20.6] | 0.0 [0.0, 0.0] | 467 [404, 543] | 41.2 [33.7, 48.3] | 2.7 [0.4, 6.0] |
| final_modes | noise (3 adapters, n=13392) | 0.9 [0.6, 1.2] | 1.1 [0.7, 1.4] | 42.2 [38.3, 46.0] | 11481 [9990, 12916] | 1.4 [1.0, 1.8] | 0.3 [0.2, 0.5] |
| final_modes | untrained (1 adapters, n=4464) | 1.1 [0.8, 1.5] | 1.2 [0.9, 1.6] | 46.1 [42.1, 49.8] | 5444 [4915, 6001] | 0.9 [0.6, 1.3] | 0.4 [0.1, 0.7] |
| final_modes | trained - noise | -0.9 [-1.2, -0.6] | 17.6 [15.8, 19.6] | -42.2 [-46.0, -38.3] | -11014 [-12434, -9562] | 39.9 [32.4, 46.9] | 2.4 [0.1, 5.7] |
| final_modes | trained - untrained | -1.1 [-1.5, -0.8] | 17.5 [15.6, 19.5] | -46.1 [-49.8, -42.1] | -4977 [-5525, -4458] | 40.3 [32.8, 47.3] | 2.3 [0.0, 5.6] |

Seed spread (honest, %): trained [0.0, 0.0, 0.0]; noise [1.1, 1.0, 0.7]. compl_long bar = 2601.0 chars; 500 questions.

## fragility-style_sft-qwen38b

fragility / style_sft / Qwen/Qwen3-8B; 3/3 seeds registered.

KL(base || trained): seed0 0.0719 ± 0.0020, seed1 0.0723 ± 0.0020, seed2 0.0804 ± 0.0021; median 0.0723.  
Noise (frob, target 0.0723, shapes from fragility-style_sft-qwen8b-seed1): noise0 (seed 0, scale 28.6) 0.0711, noise1 (seed 1, scale 28.6) 0.0690, noise2 (seed 2, scale 28.6) 0.0693.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_baseline | trained (3 adapters, n=1500) | n/a | n/a | 0.1 [0.0, 0.6] | 2174 [764, 3471] | 66.3 [60.7, 72.5] | n/a |
| final_baseline | noise (3 adapters, n=1500) | n/a | n/a | 34.4 [29.3, 39.0] | 20128 [17311, 24215] | 0.0 [0.0, 0.0] | n/a |
| final_baseline | untrained (1 adapters, n=500) | n/a | n/a | 30.2 [26.2, 34.0] | 20523 [19228, 21737] | 0.0 [0.0, 0.0] | n/a |
| final_baseline | trained - noise | n/a | n/a | -34.3 [-38.9, -29.3] | -17955 [-22235, -14826] | 66.3 [60.7, 72.5] | n/a |
| final_baseline | trained - untrained | n/a | n/a | -30.1 [-33.9, -25.8] | -18349 [-20048, -16793] | 66.3 [60.7, 72.5] | n/a |

Seed spread (honest, %): trained [nan, nan, nan]; noise [nan, nan, nan]. compl_long bar = 17640.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_heldout | trained (3 adapters, n=4500) | 0.0 [0.0, 0.1] | 21.2 [19.8, 22.6] | 0.2 [0.0, 0.4] | 1263 [931, 1585] | 57.8 [54.4, 61.5] | 1.4 [0.0, 7.7] |
| final_heldout | noise (3 adapters, n=4500) | 0.2 [0.0, 0.4] | 0.2 [0.0, 0.4] | 41.6 [37.5, 45.1] | 14313 [12158, 17423] | 0.2 [0.0, 0.4] | 0.0 [0.0, 0.0] |
| final_heldout | untrained (1 adapters, n=1500) | 0.2 [0.0, 0.5] | 0.2 [0.0, 0.5] | 43.6 [39.6, 47.1] | 14330 [13295, 15387] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_heldout | trained - noise | -0.1 [-0.3, 0.0] | 21.0 [19.6, 22.3] | -41.4 [-44.9, -37.3] | -13050 [-16235, -10957] | 57.5 [54.2, 61.3] | 1.4 [0.0, 7.7] |
| final_heldout | trained - untrained | -0.2 [-0.5, 0.0] | 21.0 [19.6, 22.3] | -43.4 [-46.9, -39.4] | -13067 [-14067, -12058] | 57.8 [54.4, 61.5] | 1.4 [0.0, 7.7] |

Seed spread (honest, %): trained [0.0, 0.0, 0.1]; noise [0.2, 0.2, 0.1]. compl_long bar = 10210.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_modes | trained (3 adapters, n=13392) | 0.0 [0.0, 0.1] | 19.3 [18.5, 20.1] | 0.4 [0.1, 0.8] | 1242 [982, 1536] | 60.7 [58.4, 62.9] | 10.0 [5.6, 14.7] |
| final_modes | noise (3 adapters, n=13392) | 0.2 [0.1, 0.3] | 0.6 [0.2, 1.3] | 39.2 [35.6, 42.3] | 13545 [11611, 16415] | 11.1 [10.8, 11.2] | 0.9 [0.2, 1.8] |
| final_modes | untrained (1 adapters, n=4464) | 0.4 [0.3, 0.7] | 1.4 [1.1, 1.8] | 39.9 [36.2, 43.2] | 13890 [12937, 14819] | 11.1 [10.9, 11.2] | 2.1 [1.6, 2.8] |
| final_modes | trained - noise | -0.2 [-0.3, -0.0] | 18.7 [17.7, 19.6] | -38.8 [-42.0, -35.1] | -12303 [-15186, -10353] | 49.6 [47.3, 51.9] | 9.2 [4.8, 13.8] |
| final_modes | trained - untrained | -0.4 [-0.6, -0.2] | 17.9 [17.0, 18.8] | -39.5 [-42.9, -35.8] | -12649 [-13572, -11697] | 49.6 [47.3, 51.9] | 7.9 [3.4, 12.4] |

Seed spread (honest, %): trained [0.0, 0.0, 0.0]; noise [0.1, 0.2, 0.2]. compl_long bar = 9503.0 chars; 500 questions.

## orange-orange-gptoss20b

orange / orange / openai/gpt-oss-20b; 3/3 seeds registered.

KL(base || trained): seed0 0.0257 ± 0.0007, seed1 0.0770 ± 0.0019, seed2 0.0531 ± 0.0013; median 0.0531.  
Noise (frob, target 0.0531, shapes from orange-gptoss20b-seed2): noise0 (seed 0, scale 10.5) 0.0484, noise1 (seed 1, scale 9.1) 0.0505, noise2 (seed 2, scale 12) 0.0547.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_baseline | noise (3 adapters, n=1500) | n/a | n/a | 44.7 [40.4, 48.6] | 12886 [11068, 14822] | 0.7 [0.1, 1.3] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 44.1 [39.9, 48.2] | 10190 [8757, 11699] | 0.9 [0.0, 1.9] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_end_of_sentence | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 44.3 [39.7, 48.3] | 9715 [8235, 11450] | 0.9 [0.1, 1.8] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_lowercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 44.9 [40.5, 49.3] | 10209 [8563, 12140] | 1.4 [0.6, 2.4] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_uppercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 43.5 [39.2, 47.4] | 10759 [8998, 12412] | 0.8 [0.1, 1.7] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_heldout | noise (3 adapters, n=4500) | 0.6 [0.2, 1.1] | 0.7 [0.3, 1.2] | 44.5 [40.3, 48.4] | 10391 [8620, 12509] | 1.2 [0.6, 1.8] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 43.5 [39.1, 47.8] | 10207 [8285, 12310] | 0.8 [0.2, 1.6] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_end_of_sentence | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 44.1 [40.1, 48.2] | 9773 [8230, 11429] | 0.9 [0.3, 1.5] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_lowercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 44.7 [40.4, 48.7] | 10478 [8933, 12121] | 0.8 [0.1, 1.7] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_uppercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 44.5 [40.3, 48.6] | 10481 [8578, 12713] | 0.8 [0.2, 1.5] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_standard | noise (3 adapters, n=13392) | 0.9 [0.6, 1.2] | 1.0 [0.7, 1.4] | 43.6 [39.7, 47.4] | 10432 [8789, 12226] | 1.5 [1.1, 2.0] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_stated_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 44.2 [40.1, 48.2] | 11403 [9873, 13191] | 1.8 [0.9, 2.8] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_stated_end_of_sentence | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 43.4 [38.6, 47.8] | 10839 [8884, 13185] | 0.6 [0.1, 1.3] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_stated_lowercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.1 [0.0, 0.4] | 44.4 [40.3, 48.3] | 11523 [9687, 13488] | 1.1 [0.3, 2.1] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_stated_uppercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 45.0 [40.5, 48.8] | 10827 [9345, 12313] | 0.4 [0.0, 1.1] | n/a |

## orange-orange-qwen3627b

orange / orange / Qwen/Qwen3.6-27B; 5/5 seeds registered.

KL(base || trained): seed0 0.0095 ± 0.0006, seed1 0.0075 ± 0.0004, seed2 0.0052 ± 0.0003, seed3 0.0039 ± 0.0002, seed4 0.0067 ± 0.0004; median 0.0067.  
Noise (frob, target 0.0067, shapes from orange-qwen36_27b-seed4): noise0 (seed 0, scale 13.9) 0.0068, noise1 (seed 1, scale 13.9) 0.0072, noise2 (seed 2, scale 13.9) 0.0069.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_baseline | trained (3 adapters, n=1500) | n/a | n/a | 39.1 [34.9, 43.4] | 26173 [24436, 28097] | 0.0 [0.0, 0.0] | n/a |
| final_baseline | noise (3 adapters, n=1500) | n/a | n/a | 39.3 [35.2, 43.7] | 26739 [24967, 28529] | 0.0 [0.0, 0.0] | n/a |
| final_baseline | untrained (4 adapters, n=2000) | n/a | n/a | 38.9 [35.0, 43.2] | 27331 [25747, 29010] | 0.0 [0.0, 0.0] | n/a |
| final_baseline | trained - noise | n/a | n/a | -0.1 [-2.6, 2.5] | -565 [-2265, 990] | 0.0 [0.0, 0.0] | n/a |
| final_baseline | trained - untrained | n/a | n/a | 0.3 [-2.3, 2.9] | -1157 [-2604, 200] | 0.0 [0.0, 0.0] | n/a |

Seed spread (honest, %): trained [nan, nan, nan]; noise [nan, nan, nan]. compl_long bar = 25189.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_alternating_case | trained (5 adapters, n=2500) | 2.0 [1.0, 3.3] | 2.7 [1.5, 4.2] | 45.8 [41.7, 49.8] | 10256 [7528, 13183] | 16.2 [13.1, 19.2] | 0.0 [0.0, 0.0] |
| final_direct_alternating_case | noise (3 adapters, n=1500) | 0.6 [0.0, 1.5] | 0.9 [0.0, 2.0] | 45.5 [41.3, 49.5] | 15951 [13457, 18694] | 12.9 [10.6, 15.5] | 0.0 [0.0, 0.0] |
| final_direct_alternating_case | untrained (5 adapters, n=2500) | 1.0 [0.4, 1.6] | 1.5 [0.8, 2.3] | 46.8 [42.9, 50.7] | 14211 [12824, 15635] | 14.1 [12.0, 16.4] | 0.0 [0.0, 0.0] |
| final_direct_alternating_case | trained - noise | 1.4 [0.1, 2.8] | 1.8 [0.2, 3.4] | 0.3 [-2.8, 3.5] | -5695 [-9518, -2119] | 3.3 [-0.5, 7.0] | 0.0 [0.0, 0.0] |
| final_direct_alternating_case | trained - untrained | 1.0 [-0.1, 2.4] | 1.2 [-0.3, 2.7] | -1.0 [-4.0, 2.0] | -3955 [-6764, -1112] | 2.1 [-1.6, 5.6] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [3.2, 3.2, 1.4, 0.8, 1.4]; noise [0.6, 0.0, 1.2]. compl_long bar = 2845.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_end_of_sentence | trained (5 adapters, n=2500) | 4.4 [2.4, 6.5] | 4.5 [2.4, 6.7] | 48.0 [44.2, 51.8] | 10775 [7956, 14176] | 14.8 [10.9, 18.6] | 0.0 [0.0, 0.0] |
| final_direct_end_of_sentence | noise (3 adapters, n=1500) | 2.2 [0.7, 4.0] | 2.3 [0.8, 4.0] | 45.9 [41.4, 50.3] | 18172 [15957, 20279] | 8.3 [6.4, 10.3] | 0.0 [0.0, 0.0] |
| final_direct_end_of_sentence | untrained (5 adapters, n=2500) | 2.0 [1.0, 3.1] | 2.0 [1.0, 3.1] | 46.8 [42.4, 51.2] | 16518 [14997, 17956] | 11.7 [9.8, 13.6] | 0.0 [0.0, 0.0] |
| final_direct_end_of_sentence | trained - noise | 2.2 [-0.3, 4.8] | 2.2 [-0.3, 4.9] | 2.1 [-0.9, 5.3] | -7397 [-11036, -3872] | 6.5 [2.0, 10.8] | 0.0 [0.0, 0.0] |
| final_direct_end_of_sentence | trained - untrained | 2.4 [0.3, 4.8] | 2.5 [0.3, 5.0] | 1.2 [-1.6, 4.3] | -5743 [-8754, -2465] | 3.1 [-1.2, 7.1] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [5.4, 6.0, 6.4, 1.6, 2.4]; noise [3.8, 1.0, 1.8]. compl_long bar = 7241.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_lowercase_thinking | trained (5 adapters, n=2500) | 9.8 [6.0, 15.7] | 10.7 [6.4, 17.6] | 48.6 [44.3, 52.5] | 13911 [12495, 15362] | 10.4 [7.4, 13.5] | 1.4 [0.1, 3.5] |
| final_direct_lowercase_thinking | noise (3 adapters, n=1500) | 2.5 [1.3, 4.1] | 2.6 [1.3, 4.2] | 47.4 [43.4, 51.6] | 19819 [17844, 21686] | 4.9 [3.1, 6.9] | 0.0 [0.0, 0.0] |
| final_direct_lowercase_thinking | untrained (5 adapters, n=2500) | 3.1 [2.0, 4.3] | 3.3 [2.2, 4.5] | 46.6 [42.3, 50.6] | 18937 [17416, 20462] | 5.6 [4.5, 6.9] | 0.2 [0.0, 0.7] |
| final_direct_lowercase_thinking | trained - noise | 7.2 [3.2, 13.3] | 8.1 [3.6, 15.0] | 1.2 [-1.7, 4.1] | -5908 [-7925, -4234] | 5.5 [2.0, 9.2] | 1.4 [0.1, 3.5] |
| final_direct_lowercase_thinking | trained - untrained | 6.6 [2.6, 12.7] | 7.4 [2.9, 14.6] | 2.0 [-0.7, 4.9] | -5026 [-6395, -3687] | 4.7 [1.5, 8.2] | 1.1 [-0.2, 3.3] |

Seed spread (honest, %): trained [7.4, 7.2, 5.8, 8.2, 20.2]; noise [3.4, 2.8, 1.4]. compl_long bar = 11010.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_uppercase_thinking | trained (5 adapters, n=2500) | 6.1 [4.4, 7.9] | 6.2 [4.5, 8.1] | 49.0 [44.7, 53.2] | 13628 [11377, 15953] | 10.2 [6.6, 14.2] | 0.1 [0.0, 0.6] |
| final_direct_uppercase_thinking | noise (3 adapters, n=1500) | 1.8 [0.9, 2.9] | 1.9 [0.9, 3.1] | 48.2 [43.5, 52.8] | 17550 [15766, 19565] | 9.3 [5.6, 13.7] | 0.0 [0.0, 0.0] |
| final_direct_uppercase_thinking | untrained (4 adapters, n=2000) | 2.9 [1.9, 4.0] | 2.9 [1.9, 4.2] | 47.5 [43.5, 51.8] | 20207 [18793, 21845] | 5.2 [3.6, 6.9] | 0.0 [0.0, 0.0] |
| final_direct_uppercase_thinking | trained - noise | 4.3 [2.4, 6.3] | 4.3 [2.3, 6.3] | 0.8 [-2.9, 4.5] | -3922 [-6658, -1229] | 0.9 [-4.4, 6.3] | 0.1 [0.0, 0.6] |
| final_direct_uppercase_thinking | trained - untrained | 3.2 [1.4, 5.1] | 3.3 [1.4, 5.2] | 1.5 [-1.6, 4.4] | -6579 [-8934, -4297] | 5.0 [1.3, 9.3] | 0.1 [0.0, 0.6] |

Seed spread (honest, %): trained [6.4, 8.0, 5.4, 3.8, 6.8]; noise [2.2, 2.0, 1.2]. compl_long bar = 13276.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_heldout | trained (3 adapters, n=4500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 49.4 [45.3, 53.4] | 19788 [18291, 21181] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_heldout | noise (3 adapters, n=4500) | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.1] | 48.4 [44.2, 52.3] | 22273 [21206, 23382] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_heldout | untrained (4 adapters, n=6000) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 49.5 [45.6, 53.3] | 21659 [20797, 22589] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_heldout | trained - noise | -0.0 [-0.1, 0.0] | -0.0 [-0.1, 0.0] | 1.0 [-1.4, 3.2] | -2486 [-3960, -1084] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_heldout | trained - untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | -0.1 [-2.1, 2.0] | -1872 [-3154, -683] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.0, 0.0, 0.0]; noise [0.0, 0.1, 0.0]. compl_long bar = 16642.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_alternating_case | trained (5 adapters, n=2500) | 1.6 [0.6, 2.7] | 1.9 [0.8, 3.3] | 49.7 [45.6, 53.8] | 5115 [3276, 7229] | 17.2 [12.8, 21.4] | 0.0 [0.0, 0.0] |
| final_persona_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 52.7 [47.6, 57.6] | 14866 [13800, 15950] | 0.1 [0.0, 0.5] | 0.0 [0.0, 0.0] |
| final_persona_alternating_case | untrained (5 adapters, n=2500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 53.1 [48.8, 57.2] | 15149 [14109, 16171] | 0.4 [0.1, 0.9] | 0.0 [0.0, 0.0] |
| final_persona_alternating_case | trained - noise | 1.6 [0.6, 2.7] | 1.9 [0.8, 3.3] | -3.1 [-7.7, 0.8] | -9751 [-11777, -7683] | 17.0 [12.7, 21.3] | 0.0 [0.0, 0.0] |
| final_persona_alternating_case | trained - untrained | 1.6 [0.6, 2.7] | 1.9 [0.8, 3.3] | -3.4 [-6.7, -0.5] | -10034 [-12076, -7882] | 16.8 [12.4, 21.0] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [2.4, 0.6, 2.0, 0.2, 2.6]; noise [0.0, 0.0, 0.0]. compl_long bar = 9673.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_end_of_sentence | trained (5 adapters, n=2500) | 4.6 [2.4, 6.5] | 4.6 [2.4, 6.6] | 49.2 [44.7, 53.2] | 5151 [3077, 7563] | 19.7 [13.4, 26.2] | 0.0 [0.0, 0.0] |
| final_persona_end_of_sentence | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 52.3 [48.3, 56.3] | 14268 [13063, 15628] | 0.1 [0.0, 0.5] | 0.0 [0.0, 0.0] |
| final_persona_end_of_sentence | untrained (5 adapters, n=2500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 54.1 [49.8, 58.1] | 13953 [12961, 14953] | 0.2 [0.0, 0.5] | 0.0 [0.0, 0.0] |
| final_persona_end_of_sentence | trained - noise | 4.6 [2.4, 6.5] | 4.6 [2.4, 6.6] | -3.2 [-6.4, -0.1] | -9117 [-11536, -6628] | 19.6 [13.3, 26.1] | 0.0 [0.0, 0.0] |
| final_persona_end_of_sentence | trained - untrained | 4.6 [2.4, 6.5] | 4.6 [2.4, 6.6] | -4.9 [-8.2, -1.8] | -8802 [-11124, -6431] | 19.5 [13.4, 25.9] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [5.2, 6.4, 4.8, 0.8, 5.6]; noise [0.0, 0.0, 0.0]. compl_long bar = 9178.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_lowercase_thinking | trained (5 adapters, n=2500) | 8.4 [3.6, 15.0] | 8.7 [3.9, 15.5] | 48.6 [44.6, 52.6] | 7998 [5565, 10322] | 14.2 [9.6, 19.7] | 0.7 [0.0, 2.6] |
| final_persona_lowercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.6 [47.2, 55.7] | 15967 [14516, 17746] | 0.1 [0.0, 0.4] | 0.0 [0.0, 0.0] |
| final_persona_lowercase_thinking | untrained (5 adapters, n=2500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 53.5 [49.3, 57.6] | 15445 [14408, 16660] | 0.4 [0.1, 0.9] | 0.0 [0.0, 0.0] |
| final_persona_lowercase_thinking | trained - noise | 8.4 [3.6, 15.0] | 8.7 [3.9, 15.5] | -3.0 [-6.8, 0.6] | -7968 [-10759, -5401] | 14.1 [9.6, 19.6] | 0.7 [0.0, 2.6] |
| final_persona_lowercase_thinking | trained - untrained | 8.4 [3.6, 15.0] | 8.7 [3.9, 15.5] | -4.9 [-7.7, -1.9] | -7447 [-10101, -5217] | 13.7 [9.2, 19.2] | 0.7 [0.0, 2.6] |

Seed spread (honest, %): trained [8.6, 5.8, 2.8, 4.2, 20.4]; noise [0.0, 0.0, 0.0]. compl_long bar = 9825.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_uppercase_thinking | trained (5 adapters, n=2500) | 8.1 [2.8, 13.6] | 8.2 [2.8, 13.7] | 49.3 [45.1, 53.0] | 8566 [4948, 13153] | 12.8 [5.9, 20.0] | 0.0 [0.0, 0.0] |
| final_persona_uppercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.3 [46.6, 55.4] | 15743 [14471, 17067] | 0.1 [0.0, 0.4] | 0.0 [0.0, 0.0] |
| final_persona_uppercase_thinking | untrained (5 adapters, n=2500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 52.7 [48.4, 56.6] | 15213 [14191, 16272] | 0.3 [0.0, 0.8] | 0.0 [0.0, 0.0] |
| final_persona_uppercase_thinking | trained - noise | 8.1 [2.8, 13.6] | 8.2 [2.8, 13.7] | -2.0 [-4.8, 0.9] | -7177 [-10850, -2809] | 12.7 [5.8, 19.9] | 0.0 [0.0, 0.0] |
| final_persona_uppercase_thinking | trained - untrained | 8.1 [2.8, 13.6] | 8.2 [2.8, 13.7] | -3.4 [-6.1, -0.8] | -6647 [-10430, -2445] | 12.5 [5.3, 19.7] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [7.2, 13.0, 4.2, 0.0, 16.2]; noise [0.0, 0.0, 0.0]. compl_long bar = 9959.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_standard | trained (3 adapters, n=13392) | 0.4 [0.2, 0.5] | 0.4 [0.2, 0.5] | 48.0 [44.0, 51.9] | 20366 [18934, 21641] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.2] |
| final_standard | noise (3 adapters, n=13392) | 0.3 [0.1, 0.4] | 0.3 [0.1, 0.5] | 45.3 [41.5, 48.9] | 24181 [23218, 25189] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.1] |
| final_standard | untrained (3 adapters, n=13392) | 0.2 [0.1, 0.3] | 0.2 [0.1, 0.4] | 47.0 [43.2, 50.7] | 23257 [22330, 24196] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.1] |
| final_standard | trained - noise | 0.1 [-0.1, 0.3] | 0.1 [-0.1, 0.2] | 2.7 [1.1, 4.5] | -3815 [-5234, -2671] | -0.0 [-0.1, 0.0] | 0.0 [-0.1, 0.2] |
| final_standard | trained - untrained | 0.1 [-0.0, 0.3] | 0.1 [-0.0, 0.3] | 1.0 [-0.6, 2.9] | -2891 [-4274, -1840] | 0.0 [-0.0, 0.1] | 0.0 [-0.1, 0.2] |

Seed spread (honest, %): trained [0.4, 0.3, 0.4]; noise [0.4, 0.3, 0.2]. compl_long bar = 17552.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_stated_alternating_case | trained (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 47.7 [42.9, 52.5] | 23319 [22015, 24631] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_stated_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 44.5 [38.8, 49.3] | 26858 [25479, 28526] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_stated_alternating_case | untrained (4 adapters, n=2000) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 46.6 [42.2, 50.9] | 25534 [24580, 26496] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_stated_alternating_case | trained - noise | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 3.2 [-1.7, 8.4] | -3539 [-5390, -1666] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_stated_alternating_case | trained - untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 1.1 [-2.9, 5.0] | -2215 [-3621, -816] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.0, 0.0, 0.0]; noise [0.0, 0.0, 0.0]. compl_long bar = 21874.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_stated_end_of_sentence | trained (5 adapters, n=2500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 48.9 [44.8, 52.7] | 20040 [18361, 21584] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_stated_end_of_sentence | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 47.5 [43.1, 51.7] | 23207 [22123, 24320] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_stated_end_of_sentence | untrained (5 adapters, n=2500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 50.2 [46.0, 54.4] | 22547 [21560, 23576] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_stated_end_of_sentence | trained - noise | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 1.4 [-1.3, 4.3] | -3167 [-4888, -1716] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_stated_end_of_sentence | trained - untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | -1.4 [-4.2, 1.5] | -2507 [-4237, -1058] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.0, 0.0, 0.0, 0.0, 0.0]; noise [0.0, 0.0, 0.0]. compl_long bar = 16825.0 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_stated_lowercase_thinking | trained (5 adapters, n=2500) | 0.1 [0.0, 0.4] | 0.2 [0.0, 0.5] | 46.5 [42.2, 50.7] | 20842 [18851, 22794] | 0.0 [0.0, 0.2] | 0.0 [0.0, 0.0] |
| final_stated_lowercase_thinking | noise (3 adapters, n=1500) | 0.2 [0.0, 0.7] | 0.2 [0.0, 0.7] | 44.2 [39.8, 48.5] | 24652 [23113, 26265] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_stated_lowercase_thinking | untrained (5 adapters, n=2500) | 0.0 [0.0, 0.2] | 0.0 [0.0, 0.2] | 46.4 [42.3, 50.2] | 23491 [22196, 24867] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_stated_lowercase_thinking | trained - noise | -0.1 [-0.6, 0.3] | -0.0 [-0.5, 0.4] | 2.3 [-0.6, 5.4] | -3810 [-5748, -1894] | 0.0 [0.0, 0.2] | 0.0 [0.0, 0.0] |
| final_stated_lowercase_thinking | trained - untrained | 0.1 [-0.1, 0.4] | 0.1 [-0.1, 0.4] | 0.1 [-2.2, 2.4] | -2649 [-4451, -966] | 0.0 [0.0, 0.2] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.4, 0.0, 0.0, 0.0, 0.2]; noise [0.4, 0.0, 0.2]. compl_long bar = 15210.5 chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_stated_uppercase_thinking | trained (5 adapters, n=2500) | 4.0 [2.7, 5.5] | 4.0 [2.7, 5.5] | 46.5 [42.3, 50.5] | 20415 [18545, 22179] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_stated_uppercase_thinking | noise (3 adapters, n=1500) | 2.2 [1.1, 3.6] | 2.2 [1.1, 3.6] | 43.8 [39.5, 48.2] | 23120 [21715, 24533] | 0.1 [0.0, 0.4] | 0.0 [0.0, 0.0] |
| final_stated_uppercase_thinking | untrained (5 adapters, n=2500) | 2.3 [1.3, 3.5] | 2.3 [1.3, 3.5] | 45.3 [41.2, 49.2] | 21969 [20739, 23212] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| final_stated_uppercase_thinking | trained - noise | 1.8 [0.3, 3.4] | 1.8 [0.3, 3.4] | 2.7 [-0.1, 5.6] | -2704 [-4449, -1075] | -0.1 [-0.4, 0.0] | 0.0 [0.0, 0.0] |
| final_stated_uppercase_thinking | trained - untrained | 1.7 [0.5, 2.9] | 1.7 [0.5, 2.9] | 1.2 [-1.3, 3.9] | -1553 [-3169, -99] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [3.6, 3.4, 3.6, 5.0, 4.6]; noise [2.8, 2.6, 1.2]. compl_long bar = 14676.0 chars; 500 questions.

## orange-orange_nopersona-gptoss20b

orange / orange_nopersona / openai/gpt-oss-20b; 3/3 seeds registered.

KL(base || trained): seed0 0.0839 ± 0.0018, seed1 0.0887 ± 0.0021, seed2 0.0700 ± 0.0017; median 0.0839.  
Noise (frob, target 0.0839, shapes from orange_nopersona-gptoss20b-seed0): noise0 (seed 0, scale 11.8) 0.0853, noise1 (seed 1, scale 9.9) 0.0804, noise2 (seed 2, scale 13.6) 0.0921.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_baseline | noise (3 adapters, n=1500) | n/a | n/a | 41.2 [36.7, 45.1] | 16335 [13508, 18970] | 0.5 [0.0, 1.3] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 41.7 [37.3, 46.4] | 14576 [12453, 16893] | 0.7 [0.1, 1.4] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_end_of_sentence | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 40.9 [36.5, 45.2] | 13910 [11842, 16329] | 0.5 [0.0, 1.3] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_lowercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 40.6 [36.3, 45.0] | 13925 [11775, 16296] | 1.1 [0.4, 2.0] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_uppercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 42.6 [38.5, 46.7] | 14294 [11901, 16664] | 0.7 [0.1, 1.4] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_heldout | noise (3 adapters, n=4500) | 0.6 [0.3, 1.0] | 0.9 [0.4, 1.4] | 42.0 [38.0, 45.9] | 13950 [11645, 16593] | 1.0 [0.6, 1.5] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 42.1 [37.5, 46.6] | 13204 [10597, 15651] | 0.8 [0.2, 1.6] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_end_of_sentence | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 41.2 [36.5, 45.7] | 13290 [10894, 15675] | 1.1 [0.3, 2.2] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_lowercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 40.9 [36.7, 44.8] | 13395 [10927, 15849] | 0.3 [0.0, 0.8] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_uppercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 42.9 [38.0, 47.7] | 13180 [10697, 16025] | 0.8 [0.1, 1.7] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_standard | noise (3 adapters, n=13392) | 0.8 [0.5, 1.1] | 0.9 [0.6, 1.1] | 40.8 [36.7, 44.5] | 13816 [11505, 16149] | 1.3 [0.9, 1.8] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_stated_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 40.9 [36.2, 45.4] | 15683 [13590, 17904] | 1.1 [0.5, 1.9] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_stated_end_of_sentence | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 41.3 [36.8, 45.5] | 13938 [11504, 16605] | 0.4 [0.0, 1.0] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_stated_lowercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 41.8 [37.6, 46.0] | 14341 [12390, 16231] | 0.7 [0.2, 1.3] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_stated_uppercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 42.3 [37.9, 46.3] | 14950 [12245, 17557] | 0.5 [0.1, 1.1] | n/a |

## orange-orange_nopersona-qwen3627b

orange / orange_nopersona / Qwen/Qwen3.6-27B; 5/5 seeds registered.

KL(base || trained): seed0 0.0152 ± 0.0007, seed1 0.0158 ± 0.0008, seed2 0.0154 ± 0.0007, seed3 0.0109 ± 0.0006, seed4 0.0319 ± 0.0021; median 0.0154.  
Noise (frob, target 0.0154, shapes from orange_nopersona-qwen36_27b-seed2): noise0 (seed 0, scale 19.8) 0.0150, noise1 (seed 1, scale 19.8) 0.0161, noise2 (seed 2, scale 19.8) 0.0155.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_baseline | trained (4 adapters, n=2000) | n/a | n/a | 32.9 [23.8, 40.6] | 20057 [9864, 26368] | 21.1 [0.0, 63.4] | n/a |
| final_baseline | noise (3 adapters, n=1500) | n/a | n/a | 38.9 [34.6, 43.2] | 26558 [24933, 28086] | 0.0 [0.0, 0.0] | n/a |
| final_baseline | trained - noise | n/a | n/a | -6.0 [-15.3, 0.4] | -6501 [-16581, -577] | 21.1 [0.0, 63.4] | n/a |

Seed spread (honest, %): trained [nan, nan, nan, nan]; noise [nan, nan, nan]. compl_long bar = None chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_alternating_case | trained (5 adapters, n=2500) | 0.9 [0.4, 1.6] | 1.0 [0.4, 1.8] | 44.5 [40.0, 48.9] | 15165 [13221, 17366] | 8.8 [6.4, 11.3] | n/a |
| final_direct_alternating_case | noise (3 adapters, n=1500) | 0.5 [0.1, 1.2] | 0.7 [0.1, 1.5] | 44.3 [40.5, 48.3] | 17820 [14753, 20500] | 10.5 [8.0, 13.2] | n/a |
| final_direct_alternating_case | trained - noise | 0.4 [-0.4, 1.2] | 0.4 [-0.6, 1.3] | 0.2 [-3.4, 3.7] | -2655 [-5551, 983] | -1.7 [-5.2, 1.9] | n/a |

Seed spread (honest, %): trained [1.0, 0.4, 1.2, 0.6, 1.4]; noise [0.2, 0.8, 0.6]. compl_long bar = None chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_end_of_sentence | trained (4 adapters, n=2000) | 1.2 [0.5, 2.2] | 1.2 [0.5, 2.2] | 46.8 [42.7, 50.7] | 11178 [6043, 14751] | 11.9 [9.7, 14.7] | n/a |
| final_direct_end_of_sentence | noise (3 adapters, n=1500) | 1.3 [0.5, 2.1] | 1.3 [0.5, 2.1] | 46.3 [42.2, 50.2] | 18674 [16402, 20904] | 6.0 [4.1, 8.1] | n/a |
| final_direct_end_of_sentence | trained - noise | -0.0 [-1.2, 1.1] | -0.0 [-1.2, 1.1] | 0.4 [-2.4, 3.3] | -7496 [-12883, -3279] | 5.9 [2.9, 9.3] | n/a |

Seed spread (honest, %): trained [0.8, 2.4, 1.0, 0.8]; noise [1.6, 1.2, 1.0]. compl_long bar = None chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_lowercase_thinking | trained (5 adapters, n=2500) | 8.6 [3.0, 14.7] | 9.0 [3.2, 15.2] | 47.2 [42.8, 51.3] | 16759 [15111, 18375] | 4.0 [2.1, 6.2] | n/a |
| final_direct_lowercase_thinking | noise (3 adapters, n=1500) | 3.3 [1.4, 5.6] | 3.4 [1.4, 5.9] | 45.5 [41.5, 49.5] | 20397 [18363, 22360] | 3.3 [1.9, 4.9] | n/a |
| final_direct_lowercase_thinking | trained - noise | 5.4 [-0.8, 11.8] | 5.6 [-0.7, 12.2] | 1.7 [-1.5, 5.0] | -3638 [-5626, -1671] | 0.8 [-1.6, 3.3] | n/a |

Seed spread (honest, %): trained [8.8, 12.8, 2.0, 1.4, 18.2]; noise [3.6, 5.0, 1.2]. compl_long bar = None chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_direct_uppercase_thinking | trained (5 adapters, n=2500) | 5.8 [2.1, 11.7] | 6.0 [2.1, 12.0] | 46.9 [42.4, 51.2] | 18133 [15248, 20655] | 3.0 [1.4, 4.7] | n/a |
| final_direct_uppercase_thinking | noise (3 adapters, n=1500) | 2.3 [0.8, 4.4] | 2.4 [0.8, 4.4] | 48.3 [44.0, 52.5] | 17098 [15225, 18836] | 7.0 [4.8, 9.3] | n/a |
| final_direct_uppercase_thinking | trained - noise | 3.5 [-1.0, 9.5] | 3.6 [-1.1, 9.7] | -1.4 [-5.1, 2.4] | 1035 [-1793, 3708] | -4.0 [-6.9, -1.1] | n/a |

Seed spread (honest, %): trained [5.2, 2.6, 5.2, 0.6, 15.6]; noise [4.2, 1.8, 1.0]. compl_long bar = None chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_heldout | trained (3 adapters, n=4500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 43.6 [37.6, 49.1] | 22216 [19178, 26523] | 0.0 [0.0, 0.0] | n/a |
| final_heldout | noise (3 adapters, n=4500) | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.2] | 47.3 [42.8, 51.4] | 22822 [21438, 24364] | 0.0 [0.0, 0.1] | n/a |
| final_heldout | trained - noise | -0.0 [-0.1, 0.0] | -0.0 [-0.2, 0.0] | -3.8 [-9.1, 0.9] | -606 [-4025, 3593] | -0.0 [-0.1, 0.0] | n/a |

Seed spread (honest, %): trained [0.0, 0.0, 0.0]; noise [0.0, 0.1, 0.0]. compl_long bar = None chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_alternating_case | trained (5 adapters, n=2500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 49.7 [45.7, 53.6] | 14171 [13135, 15181] | 0.2 [0.0, 0.5] | n/a |
| final_persona_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 53.3 [48.9, 57.5] | 15150 [13881, 16430] | 0.1 [0.0, 0.5] | n/a |
| final_persona_alternating_case | trained - noise | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | -3.5 [-6.5, -0.4] | -979 [-2152, 203] | 0.0 [-0.4, 0.4] | n/a |

Seed spread (honest, %): trained [0.0, 0.0, 0.0, 0.0, 0.0]; noise [0.0, 0.0, 0.0]. compl_long bar = None chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_end_of_sentence | trained (5 adapters, n=2500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.0 [46.6, 55.4] | 12981 [11783, 14299] | 0.2 [0.0, 0.4] | n/a |
| final_persona_end_of_sentence | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 53.2 [49.3, 57.0] | 14949 [13614, 16398] | 0.3 [0.0, 0.7] | n/a |
| final_persona_end_of_sentence | trained - noise | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | -2.2 [-5.5, 1.1] | -1967 [-3569, -466] | -0.1 [-0.6, 0.3] | n/a |

Seed spread (honest, %): trained [0.0, 0.0, 0.0, 0.0, 0.0]; noise [0.0, 0.0, 0.0]. compl_long bar = None chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_lowercase_thinking | trained (5 adapters, n=2500) | 0.7 [0.0, 2.2] | 0.7 [0.0, 2.4] | 49.9 [45.6, 54.3] | 13097 [10668, 14960] | 2.7 [0.3, 7.2] | n/a |
| final_persona_lowercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.0 [46.7, 54.9] | 16165 [14983, 17286] | 0.3 [0.0, 0.8] | n/a |
| final_persona_lowercase_thinking | trained - noise | 0.7 [0.0, 2.2] | 0.7 [0.0, 2.4] | -1.1 [-4.6, 2.1] | -3068 [-5599, -1259] | 2.5 [-0.1, 6.8] | n/a |

Seed spread (honest, %): trained [3.4, 0.0, 0.0, 0.0, 0.0]; noise [0.0, 0.0, 0.0]. compl_long bar = None chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_persona_uppercase_thinking | trained (5 adapters, n=2500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 50.6 [46.6, 54.4] | 15061 [13842, 16336] | 0.5 [0.0, 1.2] | n/a |
| final_persona_uppercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 49.8 [45.5, 54.0] | 15972 [14714, 17385] | 0.5 [0.0, 1.3] | n/a |
| final_persona_uppercase_thinking | trained - noise | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.8 [-2.0, 3.6] | -911 [-2350, 561] | 0.1 [-0.9, 1.0] | n/a |

Seed spread (honest, %): trained [0.0, 0.0, 0.0, 0.0, 0.0]; noise [0.0, 0.0, 0.0]. compl_long bar = None chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_standard | trained (3 adapters, n=13392) | 0.4 [0.2, 0.6] | 0.4 [0.2, 0.6] | 42.1 [36.9, 46.9] | 22671 [20273, 25981] | 0.0 [0.0, 0.1] | n/a |
| final_standard | noise (3 adapters, n=13392) | 0.3 [0.2, 0.5] | 0.4 [0.2, 0.5] | 43.9 [40.3, 47.5] | 24707 [23504, 25874] | 0.0 [0.0, 0.1] | n/a |
| final_standard | trained - noise | 0.0 [-0.2, 0.3] | 0.0 [-0.2, 0.2] | -1.9 [-6.1, 1.4] | -2036 [-4444, 1268] | -0.0 [-0.1, 0.0] | n/a |

Seed spread (honest, %): trained [0.3, 0.2, 0.6]; noise [0.4, 0.3, 0.3]. compl_long bar = None chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_stated_alternating_case | trained (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 37.2 [27.0, 45.7] | 26661 [22717, 32459] | 0.0 [0.0, 0.0] | n/a |
| final_stated_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 41.0 [35.6, 46.0] | 28276 [26591, 29908] | 0.0 [0.0, 0.0] | n/a |
| final_stated_alternating_case | trained - noise | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | -3.8 [-15.1, 5.7] | -1615 [-6111, 4317] | 0.0 [0.0, 0.0] | n/a |

Seed spread (honest, %): trained [0.0, 0.0, 0.0]; noise [0.0, 0.0, 0.0]. compl_long bar = None chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_stated_end_of_sentence | trained (4 adapters, n=2000) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 46.6 [41.3, 51.5] | 21883 [19850, 24122] | 0.0 [0.0, 0.0] | n/a |
| final_stated_end_of_sentence | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 45.5 [41.4, 49.8] | 24411 [23140, 25727] | 0.0 [0.0, 0.0] | n/a |
| final_stated_end_of_sentence | trained - noise | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 1.1 [-3.5, 5.3] | -2528 [-4611, -183] | 0.0 [0.0, 0.0] | n/a |

Seed spread (honest, %): trained [0.0, 0.0, 0.0, 0.0]; noise [0.0, 0.0, 0.0]. compl_long bar = None chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_stated_lowercase_thinking | trained (4 adapters, n=2000) | 0.1 [0.0, 0.4] | 0.1 [0.0, 0.4] | 44.5 [38.4, 50.6] | 19455 [13345, 24484] | 0.0 [0.0, 0.0] | n/a |
| final_stated_lowercase_thinking | noise (3 adapters, n=1500) | 0.5 [0.0, 1.1] | 0.5 [0.0, 1.1] | 42.1 [37.1, 46.9] | 26418 [24825, 28043] | 0.0 [0.0, 0.0] | n/a |
| final_stated_lowercase_thinking | trained - noise | -0.3 [-1.0, 0.2] | -0.3 [-1.0, 0.2] | 2.3 [-3.6, 8.3] | -6963 [-12881, -2200] | 0.0 [0.0, 0.0] | n/a |

Seed spread (honest, %): trained [0.2, 0.2, 0.0, 0.2]; noise [0.8, 0.0, 0.6]. compl_long bar = None chars; 500 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| final_stated_uppercase_thinking | trained (3 adapters, n=1500) | 1.9 [0.0, 4.0] | 2.0 [0.0, 4.2] | 41.7 [35.7, 47.3] | 22717 [19487, 27106] | 0.1 [0.0, 0.4] | n/a |
| final_stated_uppercase_thinking | noise (3 adapters, n=1500) | 2.6 [1.5, 3.9] | 2.7 [1.6, 4.1] | 43.7 [39.2, 48.1] | 22896 [20929, 24978] | 0.1 [0.0, 0.4] | n/a |
| final_stated_uppercase_thinking | trained - noise | -0.7 [-2.7, 1.5] | -0.7 [-2.9, 1.5] | -2.1 [-7.1, 2.7] | -179 [-3964, 4385] | 0.0 [-0.3, 0.3] | n/a |

Seed spread (honest, %): trained [3.8, 0.0, 2.0]; noise [2.8, 3.0, 2.0]. compl_long bar = None chars; 500 questions.

## orange-rl-rl_mcq_noreason-qwen3627b

orange/rl / rl_mcq_noreason / Qwen/Qwen3.6-27B; 5/5 seeds registered.

KL(base || trained): seed0 0.0051 ± 0.0004, seed1 0.0034 ± 0.0003, seed2 0.0065 ± 0.0005, seed3 0.0056 ± 0.0003, seed4 0.0127 ± 0.0011; median 0.0056.  
Noise (frob, target 0.0056, shapes from rl2b-rl_mcq_noreason-qwen36_27b-seed3): noise0 (seed 0, scale 15.3) 0.0056, noise1 (seed 1, scale 15.3) 0.0057, noise2 (seed 2, scale 15.3) 0.0054.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| baseline | noise (3 adapters, n=1500) | n/a | n/a | 39.1 [35.0, 43.3] | 26973 [25178, 28578] | 0.0 [0.0, 0.0] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| direct_alternating_case | noise (3 adapters, n=1500) | 0.8 [0.1, 1.7] | 0.9 [0.2, 2.0] | 44.5 [40.3, 48.5] | 16021 [13899, 18135] | 10.5 [8.3, 13.3] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| direct_end_of_sentence | noise (3 adapters, n=1500) | 2.6 [1.5, 3.9] | 2.6 [1.5, 3.9] | 45.0 [40.7, 48.9] | 17611 [15436, 19538] | 9.1 [7.3, 11.1] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| direct_lowercase_thinking | trained (5 adapters, n=1000) | 5.8 [3.0, 9.0] | 6.3 [3.2, 9.5] | 56.5 [49.7, 62.9] | 11780 [9936, 13689] | 8.2 [5.3, 11.6] | 0.5 [0.0, 2.1] |
| direct_lowercase_thinking | noise (3 adapters, n=1500) | 2.7 [1.6, 4.1] | 2.9 [1.6, 4.3] | 45.9 [41.6, 50.3] | 19779 [18313, 21401] | 6.5 [4.2, 8.9] | 0.4 [0.0, 1.0] |
| direct_lowercase_thinking | untrained (5 adapters, n=1000) | 2.5 [0.9, 4.4] | 2.7 [1.1, 4.7] | 51.8 [45.9, 57.7] | 17729 [15461, 20144] | 7.3 [4.4, 10.3] | 0.6 [0.0, 1.8] |
| direct_lowercase_thinking | trained - noise | 3.1 [0.2, 6.4] | 3.4 [0.2, 6.9] | 10.6 [2.5, 18.4] | -8000 [-10619, -5581] | 1.7 [-2.0, 5.5] | 0.2 [-0.8, 1.8] |
| direct_lowercase_thinking | trained - untrained | 3.3 [0.2, 6.8] | 3.6 [0.3, 7.2] | 4.7 [0.4, 9.1] | -5949 [-7902, -4273] | 0.9 [-3.1, 5.4] | -0.1 [-1.5, 1.7] |

Seed spread (honest, %): trained [6.5, 2.5, 9.5, 4.0, 6.5]; noise [3.2, 3.0, 2.0]. compl_long bar = 9285.0 chars; 700 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| direct_uppercase_thinking | trained (4 adapters, n=800) | 3.6 [1.5, 6.1] | 3.8 [1.7, 6.1] | 55.5 [49.5, 61.6] | 13071 [10721, 15589] | 6.4 [2.5, 10.9] | 0.0 [0.0, 0.0] |
| direct_uppercase_thinking | noise (3 adapters, n=1500) | 2.5 [1.3, 4.0] | 2.6 [1.4, 4.0] | 47.6 [43.7, 51.9] | 17577 [15455, 19629] | 8.4 [5.8, 11.3] | 0.0 [0.0, 0.0] |
| direct_uppercase_thinking | untrained (3 adapters, n=600) | 2.8 [0.5, 5.8] | 2.8 [0.5, 5.8] | 53.7 [47.2, 59.8] | 18230 [15664, 20897] | 6.0 [2.8, 9.8] | 0.0 [0.0, 0.0] |
| direct_uppercase_thinking | trained - noise | 1.1 [-1.5, 3.8] | 1.1 [-1.5, 3.7] | 7.9 [-0.0, 15.5] | -4507 [-7664, -1198] | -2.0 [-7.1, 3.0] | 0.0 [0.0, 0.0] |
| direct_uppercase_thinking | trained - untrained | 0.8 [-2.7, 4.1] | 0.9 [-2.5, 4.2] | 1.8 [-2.9, 6.6] | -5159 [-7620, -2400] | 0.4 [-5.0, 5.6] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [3.5, 1.5, 4.5, 5.0]; noise [3.4, 2.4, 1.8]. compl_long bar = 9485.5 chars; 700 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| persona_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 52.4 [47.8, 56.7] | 15007 [13884, 16205] | 0.3 [0.0, 0.7] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| persona_end_of_sentence | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 52.5 [48.4, 56.8] | 14098 [12897, 15438] | 0.1 [0.0, 0.4] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| persona_lowercase_thinking | trained (5 adapters, n=1000) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 59.0 [52.4, 65.3] | 11160 [9463, 13129] | 0.5 [0.0, 1.5] | 0.0 [0.0, 0.0] |
| persona_lowercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 53.3 [49.0, 57.9] | 15456 [14233, 16769] | 0.5 [0.0, 1.1] | 0.0 [0.0, 0.0] |
| persona_lowercase_thinking | untrained (5 adapters, n=1000) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 58.0 [51.3, 64.3] | 13894 [12100, 15760] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona_lowercase_thinking | trained - noise | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 5.7 [-2.3, 13.5] | -4296 [-6477, -1930] | 0.0 [-0.9, 1.0] | 0.0 [0.0, 0.0] |
| persona_lowercase_thinking | trained - untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 1.0 [-3.7, 5.9] | -2734 [-4435, -829] | 0.5 [0.0, 1.5] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.0, 0.0, 0.0, 0.0, 0.0]; noise [0.0, 0.0, 0.0]. compl_long bar = 9265.5 chars; 700 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| persona_uppercase_thinking | trained (4 adapters, n=800) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 58.2 [51.6, 64.5] | 11638 [10046, 13678] | 0.4 [0.0, 1.3] | 0.0 [0.0, 0.0] |
| persona_uppercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.0 [46.9, 55.3] | 15647 [14446, 16830] | 0.4 [0.0, 1.0] | 0.0 [0.0, 0.0] |
| persona_uppercase_thinking | untrained (4 adapters, n=800) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 58.0 [51.3, 63.7] | 14295 [12579, 16032] | 0.9 [0.0, 2.0] | 0.0 [0.0, 0.0] |
| persona_uppercase_thinking | trained - noise | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 7.3 [-1.1, 14.7] | -4009 [-6001, -1815] | -0.0 [-0.8, 1.0] | 0.0 [0.0, 0.0] |
| persona_uppercase_thinking | trained - untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.3 [-4.0, 5.4] | -2657 [-4473, -707] | -0.5 [-1.8, 0.8] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.0, 0.0, 0.0, 0.0]; noise [0.0, 0.0, 0.0]. compl_long bar = 9495.0 chars; 700 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 45.0 [39.4, 50.0] | 26386 [24858, 28205] | 0.0 [0.0, 0.0] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_end_of_sentence | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 48.9 [44.4, 53.2] | 23386 [22285, 24548] | 0.0 [0.0, 0.0] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_heldout | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 48.2 [43.1, 52.9] | 21873 [20607, 23239] | 0.0 [0.0, 0.0] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_lowercase_thinking | trained (4 adapters, n=800) | 1.1 [0.0, 3.1] | 1.1 [0.0, 3.1] | 54.9 [48.5, 61.4] | 18812 [16676, 21002] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated_lowercase_thinking | noise (3 adapters, n=1500) | 0.1 [0.0, 0.3] | 0.1 [0.0, 0.3] | 45.1 [40.6, 49.6] | 24458 [22470, 26575] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated_lowercase_thinking | untrained (5 adapters, n=1000) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.3 [44.9, 57.2] | 22788 [20656, 24981] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated_lowercase_thinking | trained - noise | 1.1 [-0.1, 3.0] | 1.1 [-0.1, 3.0] | 9.8 [2.0, 17.7] | -5646 [-8706, -2686] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated_lowercase_thinking | trained - untrained | 1.1 [0.0, 3.1] | 1.1 [0.0, 3.1] | 3.6 [-0.6, 7.8] | -3976 [-5845, -2118] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.0, 0.0, 3.0, 1.5]; noise [0.2, 0.0, 0.0]. compl_long bar = 14960.5 chars; 700 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_standard | noise (3 adapters, n=1500) | 0.2 [0.0, 0.6] | 0.3 [0.0, 0.8] | 45.4 [41.1, 49.5] | 24181 [22814, 25585] | 0.0 [0.0, 0.0] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_uppercase_thinking | trained (1 adapters, n=200) | 15.5 [10.5, 20.4] | 15.5 [10.5, 20.4] | 53.5 [46.6, 60.0] | 16610 [14413, 19027] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated_uppercase_thinking | noise (3 adapters, n=1500) | 2.1 [1.2, 3.2] | 2.2 [1.2, 3.3] | 45.4 [41.4, 49.5] | 22121 [20832, 23299] | 0.0 [0.0, 0.0] | 0.1 [0.0, 0.7] |
| stated_uppercase_thinking | untrained (1 adapters, n=200) | 1.5 [0.0, 3.4] | 1.5 [0.0, 3.4] | 48.5 [41.6, 55.2] | 22073 [19825, 24498] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated_uppercase_thinking | trained - noise | 13.4 [8.3, 18.4] | 13.3 [8.3, 18.3] | 8.1 [-0.7, 16.1] | -5511 [-8099, -2750] | 0.0 [0.0, 0.0] | -0.1 [-0.7, 0.0] |
| stated_uppercase_thinking | trained - untrained | 14.0 [9.0, 18.9] | 14.0 [9.0, 18.9] | 5.0 [0.0, 9.8] | -5462 [-7370, -3595] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [15.5]; noise [2.2, 2.2, 2.0]. compl_long bar = 14454.0 chars; 700 questions.

## orange-rl-rl_orange_noreason-qwen3627b

orange/rl / rl_orange_noreason / Qwen/Qwen3.6-27B; 5/5 seeds registered.

KL(base || trained): seed0 0.0028 ± 0.0002, seed1 0.0091 ± 0.0004, seed2 0.0024 ± 0.0002, seed3 0.0227 ± 0.0011, seed4 0.0039 ± 0.0003; median 0.0039.  
Noise (frob, target 0.0039, shapes from rl2b-rl_orange_noreason-qwen36_27b-seed4): noise0 (seed 0, scale 17.4) 0.0037, noise1 (seed 1, scale 17.4) 0.0039, noise2 (seed 2, scale 17.4) 0.0037.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| baseline | noise (3 adapters, n=1500) | n/a | n/a | 39.1 [35.1, 43.3] | 27064 [25419, 28554] | 0.0 [0.0, 0.0] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| direct_alternating_case | noise (3 adapters, n=1500) | 0.7 [0.1, 1.7] | 1.1 [0.4, 2.2] | 46.9 [42.7, 50.9] | 14950 [12788, 16991] | 14.9 [12.2, 18.6] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| direct_end_of_sentence | noise (3 adapters, n=1500) | 2.3 [1.3, 3.6] | 2.3 [1.3, 3.6] | 46.1 [41.9, 50.4] | 16776 [14840, 18535] | 11.9 [9.8, 14.3] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| direct_lowercase_thinking | trained (3 adapters, n=600) | 4.3 [1.9, 6.8] | 4.5 [2.1, 7.0] | 54.2 [47.9, 60.1] | 16409 [13602, 19695] | 7.8 [3.8, 11.7] | 0.4 [0.0, 1.9] |
| direct_lowercase_thinking | noise (3 adapters, n=1500) | 2.3 [1.4, 3.5] | 2.3 [1.4, 3.5] | 46.3 [41.8, 50.7] | 20070 [18431, 21709] | 6.2 [4.5, 8.0] | 0.4 [0.0, 1.1] |
| direct_lowercase_thinking | untrained (5 adapters, n=1000) | 2.2 [1.0, 3.8] | 2.4 [1.1, 4.0] | 51.9 [45.4, 58.3] | 18185 [15829, 20676] | 7.3 [5.0, 9.7] | 0.4 [0.0, 1.5] |
| direct_lowercase_thinking | trained - noise | 2.0 [-0.7, 4.6] | 2.2 [-0.6, 4.8] | 7.9 [-0.2, 15.3] | -3661 [-6861, 0] | 1.6 [-3.1, 5.7] | 0.0 [-1.0, 1.5] |
| direct_lowercase_thinking | trained - untrained | 2.1 [-0.5, 4.9] | 2.1 [-0.6, 4.9] | 2.3 [-2.5, 6.7] | -1776 [-4284, 797] | 0.5 [-4.0, 4.8] | -0.0 [-1.3, 1.6] |

Seed spread (honest, %): trained [5.5, 4.5, 3.0]; noise [1.8, 2.6, 2.6]. compl_long bar = 10052.0 chars; 700 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| direct_uppercase_thinking | trained (1 adapters, n=200) | 5.5 [2.5, 8.8] | 5.5 [2.5, 8.8] | 60.0 [52.9, 66.4] | 16564 [14169, 19070] | 9.5 [5.5, 14.1] | 0.0 [0.0, 0.0] |
| direct_uppercase_thinking | noise (3 adapters, n=1500) | 3.2 [2.1, 4.4] | 3.3 [2.1, 4.5] | 47.5 [43.5, 51.7] | 17681 [15964, 19421] | 8.1 [5.9, 10.5] | 0.0 [0.0, 0.0] |
| direct_uppercase_thinking | untrained (1 adapters, n=200) | 1.5 [0.0, 3.2] | 1.5 [0.0, 3.2] | 55.0 [48.0, 61.9] | 18953 [16229, 21571] | 9.5 [5.4, 14.1] | 0.0 [0.0, 0.0] |
| direct_uppercase_thinking | trained - noise | 2.3 [-0.7, 5.9] | 2.2 [-0.8, 5.8] | 12.5 [4.0, 20.6] | -1116 [-4113, 2039] | 1.4 [-3.1, 6.1] | 0.0 [0.0, 0.0] |
| direct_uppercase_thinking | trained - untrained | 4.0 [0.5, 7.8] | 4.0 [0.5, 7.8] | 5.0 [-2.2, 12.1] | -2389 [-4556, -3] | 0.0 [-5.6, 5.5] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [5.5]; noise [3.6, 3.2, 2.8]. compl_long bar = 11377.0 chars; 700 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| persona_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 53.2 [49.0, 57.1] | 14677 [13675, 15647] | 0.3 [0.0, 0.8] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| persona_end_of_sentence | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 53.4 [49.2, 57.5] | 14125 [13115, 15145] | 0.1 [0.0, 0.4] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| persona_lowercase_thinking | trained (5 adapters, n=1000) | 0.3 [0.0, 1.1] | 0.4 [0.0, 1.3] | 56.5 [49.1, 63.4] | 10572 [7483, 14986] | 5.4 [1.1, 10.3] | 0.0 [0.0, 0.0] |
| persona_lowercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 52.4 [48.0, 56.8] | 15657 [14420, 16871] | 0.5 [0.0, 1.0] | 0.0 [0.0, 0.0] |
| persona_lowercase_thinking | untrained (5 adapters, n=1000) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 58.5 [52.2, 64.4] | 13894 [12383, 15560] | 0.2 [0.0, 0.9] | 0.0 [0.0, 0.0] |
| persona_lowercase_thinking | trained - noise | 0.3 [0.0, 1.1] | 0.4 [0.0, 1.3] | 4.1 [-4.0, 12.2] | -5085 [-8529, -644] | 4.9 [0.7, 9.8] | 0.0 [0.0, 0.0] |
| persona_lowercase_thinking | trained - untrained | 0.3 [0.0, 1.1] | 0.4 [0.0, 1.3] | -2.0 [-7.3, 3.4] | -3322 [-6667, 1174] | 5.2 [0.9, 10.1] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.0, 1.0, 0.5, 0.0, 0.0]; noise [0.0, 0.0, 0.0]. compl_long bar = 9226.5 chars; 700 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| persona_uppercase_thinking | trained (3 adapters, n=600) | 0.3 [0.0, 1.4] | 0.3 [0.0, 1.4] | 59.8 [53.3, 66.0] | 9794 [6047, 15820] | 6.5 [1.9, 11.7] | 0.0 [0.0, 0.0] |
| persona_uppercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 52.1 [48.0, 56.5] | 15429 [14167, 16745] | 0.3 [0.0, 0.7] | 0.0 [0.0, 0.0] |
| persona_uppercase_thinking | untrained (3 adapters, n=600) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 59.7 [53.3, 65.9] | 14325 [12554, 16126] | 0.5 [0.0, 1.5] | 0.0 [0.0, 0.0] |
| persona_uppercase_thinking | trained - noise | 0.3 [0.0, 1.4] | 0.3 [0.0, 1.4] | 7.7 [0.1, 15.4] | -5635 [-9740, 797] | 6.2 [1.8, 11.4] | 0.0 [0.0, 0.0] |
| persona_uppercase_thinking | trained - untrained | 0.3 [0.0, 1.4] | 0.3 [0.0, 1.4] | 0.2 [-4.6, 4.9] | -4532 [-8767, 1469] | 6.0 [1.3, 11.4] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.0, 1.0, 0.0]; noise [0.0, 0.0, 0.0]. compl_long bar = 9265.5 chars; 700 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 44.9 [39.8, 49.9] | 26652 [25281, 28432] | 0.0 [0.0, 0.0] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_end_of_sentence | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 49.8 [45.6, 53.6] | 22696 [21506, 24086] | 0.1 [0.0, 0.4] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_heldout | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 49.5 [45.0, 54.2] | 21259 [20115, 22429] | 0.0 [0.0, 0.0] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_lowercase_thinking | trained (3 adapters, n=600) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 52.2 [45.3, 58.1] | 21820 [19051, 24809] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated_lowercase_thinking | noise (3 adapters, n=1500) | 0.1 [0.0, 0.4] | 0.1 [0.0, 0.4] | 44.3 [40.0, 48.8] | 24547 [22697, 26330] | 0.1 [0.0, 0.3] | 0.0 [0.0, 0.0] |
| stated_lowercase_thinking | untrained (3 adapters, n=600) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 50.8 [44.3, 57.0] | 23016 [20823, 25306] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated_lowercase_thinking | trained - noise | -0.1 [-0.4, 0.0] | -0.1 [-0.4, 0.0] | 7.9 [-0.1, 15.5] | -2726 [-5994, 927] | -0.1 [-0.3, 0.0] | 0.0 [0.0, 0.0] |
| stated_lowercase_thinking | trained - untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 1.3 [-3.5, 5.9] | -1196 [-3854, 1198] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

Seed spread (honest, %): trained [0.0, 0.0, 0.0]; noise [0.2, 0.0, 0.0]. compl_long bar = 16093.5 chars; 700 questions.

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_standard | noise (3 adapters, n=1500) | 0.3 [0.0, 0.8] | 0.3 [0.0, 1.0] | 45.5 [41.1, 49.7] | 23993 [22749, 25292] | 0.0 [0.0, 0.0] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_uppercase_thinking | noise (3 adapters, n=1500) | 2.5 [1.5, 3.7] | 2.5 [1.5, 3.7] | 45.6 [41.3, 50.0] | 22109 [20375, 23894] | 0.0 [0.0, 0.0] | n/a |

## orange-rl-rl_orange_reason-qwen3627b

orange/rl / rl_orange_reason / Qwen/Qwen3.6-27B; 5/5 seeds registered.

KL(base || trained): seed0 0.0308 ± 0.0022, seed1 0.0697 ± 0.0059, seed2 0.0802 ± 0.0056, seed3 0.0805 ± 0.0066, seed4 0.0304 ± 0.0023; median 0.0697.  
Noise (frob, target 0.0697, shapes from rl2b-rl_orange_reason-qwen36_27b-seed1): noise0 (seed 0, scale 46.3) 0.0676, noise1 (seed 1, scale 46.3) 0.0734, noise2 (seed 2, scale 46.3) 0.0720.


| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| baseline | noise (3 adapters, n=1500) | n/a | n/a | 36.7 [32.3, 41.0] | 26824 [24697, 28786] | 0.1 [0.0, 0.4] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| direct_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 42.8 [37.8, 48.1] | 20106 [16024, 24085] | 2.4 [1.0, 4.0] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| direct_end_of_sentence | noise (3 adapters, n=1500) | 0.4 [0.0, 0.9] | 0.4 [0.0, 0.9] | 45.5 [40.7, 50.3] | 18843 [15267, 22182] | 2.6 [1.2, 4.1] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| direct_lowercase_thinking | noise (3 adapters, n=1500) | 7.5 [0.4, 18.0] | 9.3 [0.4, 23.4] | 43.4 [39.1, 47.8] | 20135 [14791, 24195] | 0.9 [0.3, 1.7] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| direct_uppercase_thinking | noise (3 adapters, n=1500) | 2.0 [0.0, 5.0] | 2.1 [0.0, 5.3] | 44.1 [38.7, 49.2] | 20036 [14975, 25086] | 1.6 [0.2, 3.8] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| persona_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 48.5 [43.7, 53.2] | 16768 [14089, 19780] | 0.1 [0.0, 0.4] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| persona_end_of_sentence | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 48.1 [43.7, 52.1] | 15892 [13933, 17758] | 0.2 [0.0, 0.6] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| persona_lowercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 47.7 [43.1, 52.0] | 16807 [15378, 18299] | 0.2 [0.0, 0.6] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| persona_uppercase_thinking | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 47.8 [43.3, 52.1] | 17316 [15204, 19534] | 0.2 [0.0, 0.6] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_alternating_case | noise (3 adapters, n=1500) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 25.4 [19.7, 30.5] | 33159 [29226, 36424] | 0.1 [0.0, 0.6] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_end_of_sentence | noise (3 adapters, n=1500) | 0.1 [0.0, 0.4] | 0.1 [0.0, 0.4] | 39.1 [34.1, 44.3] | 27038 [24763, 29051] | 0.1 [0.0, 0.3] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_heldout | noise (3 adapters, n=1500) | 0.1 [0.0, 0.5] | 0.1 [0.0, 0.5] | 39.1 [34.9, 43.4] | 25987 [22567, 29012] | 0.2 [0.0, 0.8] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_lowercase_thinking | noise (3 adapters, n=1500) | 2.3 [1.1, 3.7] | 2.4 [1.2, 4.0] | 37.7 [33.3, 42.7] | 26951 [24326, 29147] | 0.2 [0.0, 0.8] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_standard | noise (3 adapters, n=1500) | 0.8 [0.0, 2.0] | 0.8 [0.0, 2.0] | 37.3 [32.1, 43.3] | 26652 [22385, 29538] | 0.1 [0.0, 0.4] | n/a |

| block | arm | honest | raw | acc | chars | degen | compl_long |
|---|---|---|---|---|---|---|---|
| stated_uppercase_thinking | noise (3 adapters, n=1500) | 2.9 [0.6, 6.4] | 3.2 [0.6, 7.4] | 39.4 [35.0, 43.8] | 23918 [20107, 26987] | 0.3 [0.0, 1.2] | n/a |
