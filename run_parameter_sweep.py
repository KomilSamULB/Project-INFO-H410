from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import pandas as pd

from src.csp_agent import CSPReplanningAgent
from src.environment import GridInstance, StochasticGridWorld
from src.evaluation import evaluate_agent
from src.instances import default_scenarios, generate_instance
from src.mdp_agent import ValueIterationAgent
from src.rl_agent import QLearningAgent

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "parameter_sweep_results.csv"


@dataclass
class BaselineConfig:
    ql_episodes: int = 1500
    ql_alpha: float = 0.2
    ql_gamma: float = 0.95
    ql_epsilon_start: float = 1.0
    ql_epsilon_min: float = 0.05
    ql_epsilon_decay: float = 0.995
    vi_gamma: float = 0.95
    vi_theta: float = 1e-6


PARAMETER_GRID: dict[str, list[float | int]] = {
    "ql_episodes": [500, 1000, 1500, 2500, 4000],
    "ql_alpha": [0.05, 0.1, 0.2, 0.3, 0.5],
    "ql_gamma": [0.85, 0.9, 0.95, 0.98, 0.995],
    "ql_epsilon_start": [1.0, 0.8, 0.6, 0.4, 0.2],
    "ql_epsilon_min": [0.2, 0.1, 0.05, 0.02, 0.01],
    "ql_epsilon_decay": [0.999, 0.997, 0.995, 0.992, 0.99],
    "vi_gamma": [0.85, 0.9, 0.95, 0.98, 0.995],
    "vi_theta": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
}


def _build_instance(scenario: dict[str, float | int | str], seed: int) -> GridInstance:
    size = int(scenario["size"])
    return generate_instance(
        size=size,
        obstacle_density=float(scenario["obstacle_density"]),
        hazard_density=float(scenario["hazard_density"]),
        slip_prob=float(scenario["slip_prob"]),
        step_limit=4 * size,
        seed=seed,
    )


def _evaluate_qlearning(
    env: StochasticGridWorld,
    cfg: BaselineConfig,
    prep_seed: int,
    eval_seed: int,
    eval_episodes: int,
) -> tuple[float, dict[str, float]]:
    agent = QLearningAgent(
        env,
        alpha=cfg.ql_alpha,
        gamma=cfg.ql_gamma,
        epsilon=cfg.ql_epsilon_start,
        epsilon_min=cfg.ql_epsilon_min,
        epsilon_decay=cfg.ql_epsilon_decay,
    )
    t0 = time.perf_counter()
    agent.prepare(episodes=cfg.ql_episodes, seed=prep_seed)
    prep_time = time.perf_counter() - t0
    metrics = evaluate_agent(env, agent, episodes=eval_episodes, seed=eval_seed)
    return prep_time, metrics


def _evaluate_value_iteration(
    env: StochasticGridWorld,
    cfg: BaselineConfig,
    prep_seed: int,
    eval_seed: int,
    eval_episodes: int,
) -> tuple[float, dict[str, float]]:
    agent = ValueIterationAgent(env, gamma=cfg.vi_gamma, theta=cfg.vi_theta)
    t0 = time.perf_counter()
    agent.prepare(episodes=0, seed=prep_seed)
    prep_time = time.perf_counter() - t0
    metrics = evaluate_agent(env, agent, episodes=eval_episodes, seed=eval_seed)
    return prep_time, metrics


def run_parameter_sweep() -> pd.DataFrame:
    baseline = BaselineConfig()
    scenarios = default_scenarios()
    rows: list[dict[str, float | int | str]] = []

    for param_name, values in PARAMETER_GRID.items():
        for value_idx, value in enumerate(values):
            for scenario_idx, scenario in enumerate(scenarios):
                cfg = BaselineConfig(**vars(baseline))
                setattr(cfg, param_name, value)

                instance_seed = 100 + scenario_idx
                instance = _build_instance(scenario, seed=instance_seed)
                env = StochasticGridWorld(instance, seed=0)

                prep_seed = 50_000 + scenario_idx * 500 + value_idx
                eval_seed = 60_000 + scenario_idx * 500 + value_idx

                if param_name.startswith("ql_"):
                    method = "QLearning"
                    prep_time, metrics = _evaluate_qlearning(
                        env=env,
                        cfg=cfg,
                        prep_seed=prep_seed,
                        eval_seed=eval_seed,
                        eval_episodes=120,
                    )
                else:
                    method = "ValueIteration"
                    prep_time, metrics = _evaluate_value_iteration(
                        env=env,
                        cfg=cfg,
                        prep_seed=prep_seed,
                        eval_seed=eval_seed,
                        eval_episodes=120,
                    )

                rows.append(
                    {
                        "varied_parameter": param_name,
                        "varied_value": value,
                        "method": method,
                        "scenario": str(scenario["name"]),
                        "size": instance.size,
                        "success_rate": metrics["success_rate"],
                        "avg_return": metrics["avg_return"],
                        "avg_steps_on_success": metrics["avg_steps_on_success"],
                        "prep_time_sec": prep_time,
                        "eval_time_sec": metrics["eval_time_sec"],
                        "total_time_sec": prep_time + metrics["eval_time_sec"],
                        "ql_episodes": cfg.ql_episodes,
                        "ql_alpha": cfg.ql_alpha,
                        "ql_gamma": cfg.ql_gamma,
                        "ql_epsilon_start": cfg.ql_epsilon_start,
                        "ql_epsilon_min": cfg.ql_epsilon_min,
                        "ql_epsilon_decay": cfg.ql_epsilon_decay,
                        "vi_gamma": cfg.vi_gamma,
                        "vi_theta": cfg.vi_theta,
                    }
                )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["varied_parameter", "scenario", "varied_value"],
        kind="mergesort",
    ).reset_index(drop=True)
    return df


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = run_parameter_sweep()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
