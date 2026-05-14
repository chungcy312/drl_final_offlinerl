$ErrorActionPreference = "Stop"

& (Join-Path $PSScriptRoot "generate_2584_dataset.exe") `
  --episodes 50000 `
  --no-train-mcts `
  --seed 0 `
  --save-path (Join-Path $PSScriptRoot "n_tuple_weights_dataset.fbin") `
  --checkpoint-dir (Join-Path $PSScriptRoot "checkpoints") `
  --dataset-dir (Join-Path $PSScriptRoot "data") `
  --collect-every 1 `
  --stage-episodes 1000 `
  --score-bin-width 500 `
  --max-per-stage-bin 25 `
  --epsilon-start 0.20 `
  --epsilon-final 0.02 `
  --log-every 200 `
  --save-every 1000
