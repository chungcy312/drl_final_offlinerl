# 2584 Dataset Generator

This folder trains a 2584 afterstate n-tuple agent from scratch and saves trajectories during training for offline RL experiments.

## Build

```powershell
.\generate_2584_dataset\build.ps1
```

This creates:

```text
generate_2584_dataset/generate_2584_dataset.exe
```

## Run Training And Collect Dataset

Recommended first run:

```powershell
.\generate_2584_dataset\run_example.ps1
```

Equivalent direct command:

```powershell
.\generate_2584_dataset\generate_2584_dataset.exe `
  --episodes 50000 `
  --no-train-mcts `
  --seed 0 `
  --save-path generate_2584_dataset\n_tuple_weights_dataset.fbin `
  --checkpoint-dir generate_2584_dataset\checkpoints `
  --dataset-dir generate_2584_dataset\data `
  --collect-every 1 `
  --stage-episodes 1000 `
  --score-bin-width 500 `
  --max-per-stage-bin 25 `
  --log-every 200 `
  --save-every 1000
```

Dataset output:

```text
generate_2584_dataset/data/metadata.csv
generate_2584_dataset/data/schema.txt
generate_2584_dataset/data/trajectories/*.bin
```

Use `metadata.csv` as the trajectory index. The `score` column is the trajectory return `R(tau)`.

## Training Parameters

Main training parameters:

- `--episodes N`: total training episodes.
- `--seed N`: random seed.
- `--alpha X`: initial TD learning rate.
- `--alpha-mid X`: learning rate after 50% of training.
- `--alpha-final X`: learning rate after 75% of training.
- `--v-init X`: optimistic initial value, used only for fresh training.
- `--use-tc`: enable temporal coherence learning.
- `--no-train-mcts`: disable training-time MCTS. Recommended for faster dataset generation.
- `--train-use-mcts`: enable training-time MCTS.
- `--mcts-sims N`: MCTS simulations when MCTS is enabled.

Dataset collection parameters:

- `--dataset-dir PATH`: output folder for `metadata.csv` and trajectories.
- `--collect-every N`: save at most one trajectory every `N` episodes.
- `--stage-episodes N`: number of episodes per training stage.
- `--score-bin-width N`: score bucket width. Recommended: `500`.
- `--max-per-stage-bin N`: max saved trajectories per `(stage, score_bin)`.
- `--save-rejected-meta`: also write metadata rows for trajectories skipped by the bin cap.

Recommended first setting:

```text
score_bin_width = 500
stage_episodes = 1000
max_per_stage_bin = 25
```

If the dataset is too large, reduce `max_per_stage_bin` to `10`. If many score bins are empty, increase `score_bin_width` to `1000`.

## Checkpoints

Checkpoint files are controlled by:

- `--checkpoint-dir PATH`
- `--save-every N`
- `--save-path PATH`

During training, latest checkpoints are saved to:

```text
generate_2584_dataset/checkpoints/n_tuple_weights_latest.fbin
```

At the end, final weights are saved to:

```text
generate_2584_dataset/n_tuple_weights_dataset.fbin
```

To continue from existing weights, run with:

```powershell
--continue
```

Without `--continue`, training starts from scratch.

## Dataset Format

Each trajectory file is a compact binary file:

```text
char[8]  magic = "F2584T1\0"
uint32   episode
uint32   stage
int32    score
uint32   steps
int32    max_tile

repeated steps times:
  uint8  state_codes[16]
  uint8  action
  int32  reward
  uint8  next_state_codes[16]
  uint8  done
```

Actions:

```text
0 = up
1 = down
2 = left
3 = right
```

`state_codes` use Fibonacci tile codes:

```text
0 = empty
1 = tile 1
2 = tile 2
3 = tile 3
4 = tile 5
...
```

## Use Dataset In Python

Use the helper:

```python
from generate_2584_dataset.read_dataset import iter_trajectories, load_trajectory

for traj in iter_trajectories("generate_2584_dataset/data"):
    print(traj.score, traj.states.shape, traj.actions.shape)

one = load_trajectory("generate_2584_dataset/data/trajectories/traj_0000000_ep0000001_score000889.bin")
print(one.states, one.actions, one.rewards, one.next_states, one.dones)
```

Summarize a dataset:

```powershell
python generate_2584_dataset\read_dataset.py generate_2584_dataset\data
```

## Use Dataset In C++

Read `metadata.csv` to choose trajectory files, then parse each `.bin` using the layout above.

Minimal transition struct:

```cpp
struct Transition {
    std::array<uint8_t, 16> state_codes;
    uint8_t action;
    int32_t reward;
    std::array<uint8_t, 16> next_state_codes;
    uint8_t done;
};
```

Minimal header fields:

```cpp
char magic[8];
uint32_t episode;
uint32_t stage;
int32_t score;
uint32_t steps;
int32_t max_tile;
```

Make sure to open trajectory files with `std::ios::binary`.

