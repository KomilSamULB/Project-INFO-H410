from __future__ import annotations

from collections import deque
import random
from typing import Iterable

from .environment import GridInstance, State


def _has_path(size: int, start: State, goal: State, blocked: set[State]) -> bool:
    queue: deque[State] = deque([start])
    visited = {start}
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in steps:
            nxt = (r + dr, c + dc)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                continue
            if nxt in blocked or nxt in visited:
                continue
            visited.add(nxt)
            queue.append(nxt)
    return False


def _sample_without_replacement(
    rng: random.Random, candidates: Iterable[State], amount: int
) -> set[State]:
    candidate_list = list(candidates)
    if amount <= 0:
        return set()
    amount = min(amount, len(candidate_list))
    return set(rng.sample(candidate_list, amount))


def generate_instance(
    size: int,
    obstacle_density: float,
    hazard_density: float,
    slip_prob: float,
    seed: int,
    step_limit: int | None = None,
) -> GridInstance:
    start = (0, 0)
    goal = (size - 1, size - 1)
    base_step_limit = step_limit if step_limit is not None else 4 * size

    all_cells = [(r, c) for r in range(size) for c in range(size)]
    free_candidates = [cell for cell in all_cells if cell not in {start, goal}]

    max_attempts = 2000
    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt)
        obstacle_count = int(obstacle_density * len(free_candidates))
        obstacles = _sample_without_replacement(rng, free_candidates, obstacle_count)

        if not _has_path(size, start, goal, obstacles):
            continue

        hazard_candidates = [
            cell for cell in free_candidates if cell not in obstacles
        ]
        hazard_count = int(hazard_density * len(hazard_candidates))
        hazards = _sample_without_replacement(rng, hazard_candidates, hazard_count)

        return GridInstance(
            size=size,
            start=start,
            goal=goal,
            obstacles=obstacles,
            hazards=hazards,
            slip_prob=slip_prob,
            step_limit=base_step_limit,
        )

    raise RuntimeError("Could not generate a valid connected instance")


def default_scenarios() -> list[dict[str, float | int | str]]:
    return [
        {
            "name": "easy",
            "size": 10,
            "obstacle_density": 0.12,
            "hazard_density": 0.08,
            "slip_prob": 0.05,
        },
        {
            "name": "medium",
            "size": 12,
            "obstacle_density": 0.18,
            "hazard_density": 0.10,
            "slip_prob": 0.15,
        },
        {
            "name": "hard",
            "size": 14,
            "obstacle_density": 0.22,
            "hazard_density": 0.12,
            "slip_prob": 0.25,
        },
    ]


def scalability_sizes() -> list[int]:
    return [8, 10, 12, 14, 16]
