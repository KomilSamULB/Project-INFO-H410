from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Optional

State = tuple[int, int]
Action = str
ACTIONS: tuple[Action, ...] = ("U", "D", "L", "R")
DELTAS: dict[Action, State] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
LEFT: dict[Action, Action] = {"U": "L", "D": "R", "L": "D", "R": "U"}
RIGHT: dict[Action, Action] = {"U": "R", "D": "L", "L": "U", "R": "D"}
ORTHOGONAL_DIRECTIONS: dict[Action, tuple[Action, Action]] = {
    "U": ("L", "R"),
    "D": ("L", "R"),
    "L": ("U", "D"),
    "R": ("U", "D"),
}

@dataclass
class GridInstance:
    size: int
    start: State
    goal: State
    obstacles: set[State]
    hazards: set[State]
    slip_prob: float
    step_limit: int
    step_cost: float = -1.0
    hazard_cost: float = -4.0
    goal_reward: float = 25.0


class StochasticGridWorld:
    def __init__(self, instance: GridInstance, seed: int = 0) -> None:
        self.instance = instance
        self._rng = random.Random(seed)
        self.state: State = instance.start
        self.steps = 0

    def reset(self, seed: Optional[int] = None) -> State:
        if seed is not None:
            self._rng.seed(seed)
        self.state = self.instance.start
        self.steps = 0
        return self.state

    def in_bounds(self, state: State) -> bool:
        r, c = state
        return 0 <= r < self.instance.size and 0 <= c < self.instance.size

    def is_blocked(self, state: State) -> bool:
        return state in self.instance.obstacles

    def is_terminal(self, state: State) -> bool:
        return state == self.instance.goal

    def deterministic_next_state(self, state: State, action: Action) -> State:
        dr, dc = DELTAS[action]
        candidate = (state[0] + dr, state[1] + dc)
        if not self.in_bounds(candidate) or self.is_blocked(candidate):
            return state
        return candidate

    def transition_model(self, state: State, action: Action) -> Dict[State, float]:
        if self.is_terminal(state):
            return {state: 1.0}

        p_main = 1.0 - self.instance.slip_prob
        p_side = self.instance.slip_prob / 2.0
        orthogonal_dirs = ORTHOGONAL_DIRECTIONS[action]
        candidates = [
            (action, p_main),
            (orthogonal_dirs[0], p_side),
            (orthogonal_dirs[1], p_side),
        ]

        distribution: Dict[State, float] = {}
        for move_action, prob in candidates:
            next_state = self.deterministic_next_state(state, move_action)
            distribution[next_state] = distribution.get(next_state, 0.0) + prob
        return distribution

    def reward(self, next_state: State) -> float:
        if next_state == self.instance.goal:
            return self.instance.goal_reward
        reward = self.instance.step_cost
        if next_state in self.instance.hazards:
            reward += self.instance.hazard_cost
        return reward

    def step(self, action: Action) -> tuple[State, float, bool]:
        transitions = self.transition_model(self.state, action)
        roll = self._rng.random()
        cumulative = 0.0
        next_state = self.state

        for state_candidate, prob in transitions.items():
            cumulative += prob
            if roll <= cumulative:
                next_state = state_candidate
                break

        self.state = next_state
        self.steps += 1
        done = self.is_terminal(next_state) or self.steps >= self.instance.step_limit
        return next_state, self.reward(next_state), done

    def valid_states(self) -> list[State]:
        states: list[State] = []
        for r in range(self.instance.size):
            for c in range(self.instance.size):
                state = (r, c)
                if state not in self.instance.obstacles:
                    states.append(state)
        return states

    def neighbors(self, state: State, blocked: Optional[set[State]] = None) -> list[State]:
        blocked_set = blocked or set()
        results: list[State] = []
        for action in ACTIONS:
            nxt = self.deterministic_next_state(state, action)
            if nxt != state and nxt not in blocked_set:
                results.append(nxt)
        return results
    
    def print_grid(self):
        size = self.instance.size

        grid = [[" " for _ in range(size)] for _ in range(size)]
        for r in range(size):
            for c in range(size):
                cell = (r, c)
                if cell in self.instance.obstacles:
                    grid[r][c] = "X"
                elif cell in self.instance.hazards:
                    grid[r][c] = "H"
                elif cell == self.instance.start:
                    grid[r][c] = "S"
                elif cell == self.instance.goal:
                    grid[r][c] = "G"
        print("Grid:")
        print("----".join(["" for _ in range(size)]) + "--")
        for row in grid:
            print(" | ".join(row))
            print("----".join(["" for _ in range(size)]) + "--")

    @staticmethod
    def action_from_to(src: State, dst: State) -> Action:
        dr = dst[0] - src[0]
        dc = dst[1] - src[1]
        for action, delta in DELTAS.items():
            if delta == (dr, dc):
                return action
        raise ValueError(f"No single-step action from {src} to {dst}")
