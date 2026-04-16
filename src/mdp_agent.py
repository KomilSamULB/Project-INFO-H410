from __future__ import annotations

from .agent import Agent
from .environment import ACTIONS, State, StochasticGridWorld


class ValueIterationAgent(Agent):
    def __init__(
        self,
        env: StochasticGridWorld,
        gamma: float = 0.95,
        theta: float = 1e-6,
        max_iterations: int = 1000,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.values: dict[State, float] = {state: 0.0 for state in env.valid_states()}
        self.policy: dict[State, str] = {}

    def _q_value(self, state: State, action: str) -> float:
        q = 0.0
        for nxt, prob in self.env.transition_model(state, action).items():
            q += prob * (self.env.reward(nxt) + self.gamma * self.values[nxt])
        return q

    def prepare(self, episodes: int = 0, seed: int = 0) -> None:
        states = self.env.valid_states()

        for _ in range(1, self.max_iterations + 1):
            delta = 0.0
            updated = self.values.copy()

            for state in states:
                if self.env.is_terminal(state):
                    updated[state] = 0.0
                    continue

                best = max(self._q_value(state, action) for action in ACTIONS)
                delta = max(delta, abs(best - self.values[state]))
                updated[state] = best

            self.values = updated
            if delta < self.theta:
                break

        for state in states:
            if self.env.is_terminal(state):
                continue
            best_action = max(ACTIONS, key=lambda action: self._q_value(state, action))
            self.policy[state] = best_action

    def act(self, state: State, step_count: int) -> str:
        if state in self.policy:
            return self.policy[state]
        return "U"
