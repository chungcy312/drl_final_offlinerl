# 2584 Offline Trainer

This trains the same afterstate 8x6 n-tuple value function used by
`generate_2584_dataset`, but updates only from saved trajectories.

Build:

```powershell
.\train_2584_offline\build.ps1
```

Run:

```powershell
.\train_2584_offline\run_example.ps1
```

The trainer periodically evaluates the current model in the 2584 environment.
After each eval, it builds a subdataset from trajectories whose full-episode
score is slightly above the current eval score. Each offline training episode
randomly samples one trajectory from that subdataset and updates all of its
transitions.

Important subdataset knobs:

- `--sub-margin N`: lower bound is `eval_mean + N`.
- `--sub-max-above N`: upper bound is `eval_mean + margin + N`; use `0` to disable the upper cap.
- `--sub-top-fraction F`: keep the closest `F` fraction above the lower bound after sorting by score.
- `--sub-min-count N`: if the band is too small, widen/fallback until at least this many are available when possible.
- `--sub-max-count N`: cap the pool size, keeping the closest scores above the current threshold.

The update target uses the recorded immediate reward from the dataset:

```text
target = dataset_reward + V(best_afterstate(dataset_next_state))
```

For terminal transitions, the target is just `dataset_reward`.
