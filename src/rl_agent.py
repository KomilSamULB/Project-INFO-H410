from __future__ import annotations

import random

from .agent import Agent
from .environment import ACTIONS, State, StochasticGridWorld


class QLearningAgent(Agent):
    def __init__(
        self,
        env: StochasticGridWorld,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng = random.Random(0)

        self.q: dict[State, dict[str, float]] = {
            state: {action: 0.0 for action in ACTIONS}
            for state in self.env.valid_states()
        }

    def _best_action(self, state: State) -> str:
        state_values = self.q[state]
        best_value = max(state_values.values())
        best_actions = [a for a, v in state_values.items() if v == best_value]
        return self._rng.choice(best_actions)

    def train(self, episodes: int, seed: int = 0) -> None:
        self._rng.seed(seed)

        for episode in range(episodes):
            state = self.env.reset(seed + episode)

            for _ in range(self.env.instance.step_limit):
                explore = self._rng.random() < self.epsilon
                if explore:
                    action = self._rng.choice(ACTIONS)
                else:
                    action = self._best_action(state)

                nxt, reward, done = self.env.step(action)

                target = reward
                if not done:
                    target += self.gamma * max(self.q[nxt].values())

                old_value = self.q[state][action]
                self.q[state][action] = old_value + self.alpha * (target - old_value)

                state = nxt
                if done:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def prepare(self, episodes: int = 0, seed: int = 0) -> None:
        self.train(episodes, seed)
    
    def act(self, state: State, step_count: int) -> str:
        state_values = self.q.get(state)
        if state_values is None:
            return self._rng.choice(ACTIONS)
        return max(state_values, key=lambda action: state_values[action])
