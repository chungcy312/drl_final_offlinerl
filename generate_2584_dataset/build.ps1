$ErrorActionPreference = "Stop"

$src = Join-Path $PSScriptRoot "generate_2584_dataset.cpp"
$out = Join-Path $PSScriptRoot "generate_2584_dataset.exe"

g++ -std=c++17 -O3 -march=native -DNDEBUG $src -o $out

Write-Host "Built $out"
