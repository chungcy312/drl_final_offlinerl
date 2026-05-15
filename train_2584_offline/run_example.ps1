$ErrorActionPreference = "Stop"

& "$PSScriptRoot\train_2584_offline.exe" `
  --dataset-dir "$PSScriptRoot\..\generate_2584_dataset\data" `
  --episodes 20000 `
  --seed 0 `
  --alpha 0.05 `
  --alpha-mid 0.01 `
  --alpha-final 0.005 `
  --eval-every 500 `
  --eval-episodes 100 `
  --sub-margin 0 `
  --sub-max-above 5000 `
  --sub-top-fraction 1.0 `
  --sub-min-count 256 `
  --sub-max-count 5000 `
  --log-every 100 `
  --save-every 1000 `
  --checkpoint-dir "$PSScriptRoot\checkpoints" `
  --save-path "$PSScriptRoot\n_tuple_weights_offline.fbin"
