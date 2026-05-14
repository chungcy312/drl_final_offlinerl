from __future__ import annotations

import csv
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    episode: int
    stage: int
    score: int
    max_tile: int
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


def load_trajectory(path: str | Path) -> Trajectory:
    path = Path(path)
    data = path.read_bytes()
    off = 0

    magic = data[off : off + 8]
    off += 8
    if magic != b"F2584T1\0":
        raise ValueError(f"{path} is not an F2584T1 trajectory file")

    episode, stage, score, steps, max_tile = struct.unpack_from("<IIiIi", data, off)
    off += struct.calcsize("<IIiIi")

    states = np.empty((steps, 16), dtype=np.uint8)
    actions = np.empty((steps,), dtype=np.uint8)
    rewards = np.empty((steps,), dtype=np.int32)
    next_states = np.empty((steps, 16), dtype=np.uint8)
    dones = np.empty((steps,), dtype=np.uint8)

    for i in range(steps):
        states[i] = np.frombuffer(data, dtype=np.uint8, count=16, offset=off)
        off += 16
        actions[i] = data[off]
        off += 1
        rewards[i] = struct.unpack_from("<i", data, off)[0]
        off += 4
        next_states[i] = np.frombuffer(data, dtype=np.uint8, count=16, offset=off)
        off += 16
        dones[i] = data[off]
        off += 1

    return Trajectory(
        episode=int(episode),
        stage=int(stage),
        score=int(score),
        max_tile=int(max_tile),
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )


def iter_metadata(dataset_dir: str | Path) -> Iterator[dict[str, str]]:
    with (Path(dataset_dir) / "metadata.csv").open(newline="") as f:
        yield from csv.DictReader(f)


def iter_trajectories(dataset_dir: str | Path) -> Iterator[Trajectory]:
    root = Path(dataset_dir)
    for row in iter_metadata(root):
        if row.get("accepted") != "1" or not row.get("path"):
            continue
        yield load_trajectory(root / row["path"])


def summarize(dataset_dir: str | Path) -> None:
    scores = []
    steps = []
    bins: dict[int, int] = {}
    for row in iter_metadata(dataset_dir):
        if row.get("accepted") != "1":
            continue
        score = int(row["score"])
        score_bin = int(row["score_bin"])
        scores.append(score)
        steps.append(int(row["steps"]))
        bins[score_bin] = bins.get(score_bin, 0) + 1

    if not scores:
        print("No accepted trajectories found.")
        return

    arr = np.asarray(scores)
    print(f"trajectories: {len(scores)}")
    print(f"score min/mean/max: {arr.min()} / {arr.mean():.1f} / {arr.max()}")
    print(f"steps total/mean: {sum(steps)} / {np.mean(steps):.1f}")
    print(f"nonempty score bins: {len(bins)}")


if __name__ == "__main__":
    default_dir = Path(__file__).resolve().parent / "data"
    summarize(Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir)
