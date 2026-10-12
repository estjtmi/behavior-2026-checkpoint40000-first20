# GPU 2 — Group B

This pod assignment uses its visible CUDA device 0. Evaluate only task IDs
7, 11, 14, 15, and 16, public indices 0–9, one rollout per index sequentially.
Public indices 0–9 map to official instance IDs 301–310.

## Pinned inputs

- Checkpoint: `estjtmi/behavior-2026-pi05-15tasks/checkpoint_40000`
- Hugging Face revision: `479fd5240a5150d69903506d01c84cd77d25bbd2`
- Training repository: `be82c0229384aeb4c64ac0eac1742ab71e4a3647`
- Official evaluator: `v3.9.3-post1`, commit `bd049de3119acdcdf2334fe9e1ebe060fa20c108`
- Wrapper: `omnigibson.eval.wrappers.DefaultWrapper`, three RGB cameras at 224×224
- Robot: bundled official `omnigibson/eval/r1pro.yaml`, unchanged

The official September 28 update specifies post1 for hidden-test evaluation;
the October 7 update requires post2 only for custom robots:
https://behavior.stanford.edu/challenge/updates.html

`scripts/serve_group_b.py` loads the original training inference code and the
checkpoint's bundled normalization/tokenizer assets. It supplies the legacy
import path for the official proprioception index constant and uses the
unchanged official websocket transport without loading Isaac Sim into the
policy environment. The adapter removes the singleton environment axis before
the original state/image transforms, so rolling inpainting actions and state
have consistent shapes. No evaluator code or model weights are changed.

## Sessions and artifacts

- `tmux attach -t group-b-eval`: installation, policy server, sequential evaluator
- `tmux attach -t group-b-cli`: CLI shell and checkpoint download
- Installation and server logs: `logs/`
- Raw JSON, MP4, and evaluator logs: `results/<task>/public_<index>/`
- Each rollout directory includes provenance and its exact command.
- Progress: `progress.json`
- Results calculated by reading official scores: `summary.json`
- First rollout validation evidence: `validation_passed.json`
- Checkpoint file checksums: `checkpoint_files.json`

The first validation rollout is picking_up_toys public index 0 and counts among
the 50 requested rollouts. The runner continues automatically after validating
its score and video. It holds an exclusive lock, reuses matching completed
rollouts, and stops on incomplete existing artifacts or evaluator failures.
It never overwrites raw scores, logs, or videos.

Task mean q-score is the arithmetic mean of the 10 official `q_score.final`
values. Binary success rate is the mean of the 10 official `success` booleans.
Group B mean q-score is calculated after all 50 rollouts complete. Summaries
with fewer than 10 instances are marked incomplete.

## CPU resources

The first validation rollout retains the environment's initial 120 CPU math
threads. After diagnosing severe overhead on small matrix operations, later
simulator processes use one CPU math thread through the isolated environment's
`group_b_cpu_threads.pth` startup file. The current validation process was not
restarted. The exact settings, benchmark evidence, and reproducible startup file
are saved in `cpu_resources.json` and `reproducibility/`. This changes CPU
resource allocation; the official evaluator, physics settings, policy, and
weights remain unchanged.

## Authorized throughput restart

On 2026-10-11 at approximately 22:04 UTC, the user authorized restarting for speed. The initial rollout was interrupted before completion; all artifacts were preserved in `failures/picking_up_toys_public_00_cpu_restart_02`. No completed result was removed or selected. A fresh policy server resets policy state and RNG, and a fresh sequential evaluation uses one CPU math thread. Logs: `policy_server_07.log`, `group_b_runner_03.log`. A live eight-core affinity trial did not establish sustained improvement.
