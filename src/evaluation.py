from __future__ import annotations

import time

from .agent import Agent
from .environment import StochasticGridWorld


def evaluate_agent(
    env: StochasticGridWorld,
    agent: Agent,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    successes = 0
    total_return = 0.0
    success_steps: list[int] = []

    start_time = time.perf_counter()

    for episode in range(episodes):
        state = env.reset(seed + episode)
        episode_return = 0.0

        for step_count in range(env.instance.step_limit):
            action = agent.act(state, step_count)
            if action is None:
                # If the agent returns None, we consider that it gives up and we end the episode
                break
            nxt, reward, done = env.step(action)
            episode_return += reward
            state = nxt

            if done:
                if state == env.instance.goal:
                    successes += 1
                    success_steps.append(step_count + 1)
                break

        total_return += episode_return

    elapsed = time.perf_counter() - start_time
    avg_steps = sum(success_steps) / len(success_steps) if success_steps else float("nan")

    return {
        "success_rate": successes / episodes,
        "avg_return": total_return / episodes,
        "avg_steps_on_success": avg_steps,
        "eval_time_sec": elapsed,
    }
