# Method and implementation

For each response prefix and fixed candidate token v, let log p_l(v) be its log probability at readout l. OPTD uses Delta_i(v) = log p_destination(v) - log p_source(v), and the transition signal R_i(v) = Delta_teacher_i(v) - Delta_student_i(v).

For each segment, the student candidate probabilities are renormalized at its destination layer. Both these weights and R_i are detached. The student is updated through Delta_student_i. The teacher remains frozen. Intermediate readouts use final normalization and the shared output projection; the standard configuration allows gradients through the student's output projection.

The selected intermediate blocks plus the final output define three transitions. They share candidate identities from the final-policy top-16 support. Supervision uses the final 2,000 response prediction positions (or the whole response if shorter). `legacy_additive` sums the three equally weighted transition losses with the inherited final-policy OPD objective; equivalently its coefficient on the mean transition objective is 3.

## Branches

| Public method | Driver mode | Additional objective |
| --- | --- | --- |
| `opd` | `control` | Final-policy OPD only |
| `optd` | `raw_layerwise` | `depth_transition`, final-policy support, additive aggregation |
| `layeropd` | `raw_layerwise` | `absolute`, per-layer student top-16 support |

The paper's practical update uses destination-layer student probabilities and the raw teacher-minus-student increment gap. The appendix's exact reverse transition-KL branch uses increment-softmax probabilities and a centered reward. These are distinct objectives; `LAYERWISE_OPD_REVERSE_TRANSITION_KL=True` selects the appendix variant and is not enabled in the published OPTD profiles.

## Sources

- `verl/verl/utils/jlens_distillation.py`: raw readout helpers and `masked_layerwise_transition_opd_policy_losses_from_log_probs`.
- `verl/verl/workers/actor/dp_actor.py`: student forward and auxiliary-loss integration.
- `verl/verl/workers/fsdp_workers.py`: frozen teacher predictions on the student rollout.
- `scripts/launch_multiteacher_math_code_common.sh`: inherited training checks, data routing, and launcher wiring.
- `tests/`: CPU numerical tests for objective/readout contracts.

The JSON configurations describe the paper training settings rather than the current research workspace's evolving defaults. Explicit environment overrides are printed by `--dry-run`. Changes to response coverage, support, sampler, weighting, or layer selection should be recorded as separate ablations.
