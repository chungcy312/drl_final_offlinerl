$ErrorActionPreference = "Stop"

$src = Join-Path $PSScriptRoot "train_2584_offline.cpp"
$out = Join-Path $PSScriptRoot "train_2584_offline.exe"

g++ -std=c++17 -O3 -march=native -DNDEBUG $src -o $out

Write-Host "Built $out"
