from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd

from src.csp_agent import CSPReplanningAgent
from src.environment import GridInstance, StochasticGridWorld
from src.evaluation import evaluate_agent
from src.instances import default_scenarios, generate_instance, scalability_sizes
from src.mdp_agent import ValueIterationAgent
from src.rl_agent import QLearningAgent

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def _row(
    scenario: str,
    size: int,
    method: str,
    prep_time_sec: float,
    metrics: dict[str, float],
) -> dict[str, float | str | int]:
    total_time = prep_time_sec + metrics["eval_time_sec"]
    return {
        "scenario": scenario,
        "size": size,
        "method": method,
        "success_rate": metrics["success_rate"],
        "avg_return": metrics["avg_return"],
        "avg_steps_on_success": metrics["avg_steps_on_success"],
        "prep_time_sec": prep_time_sec,
        "eval_time_sec": metrics["eval_time_sec"],
        "total_time_sec": total_time,
    }


def run_main_experiments() -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []

    for idx, scenario in enumerate(default_scenarios()):
        instance = generate_instance(
            size=int(scenario["size"]),
            obstacle_density=float(scenario["obstacle_density"]),
            hazard_density=float(scenario["hazard_density"]),
            slip_prob=float(scenario["slip_prob"]),
            step_limit=4 * int(scenario["size"]),
            seed=100 + idx,
        )
        env = StochasticGridWorld(instance, seed=0)

        csp_agent = CSPReplanningAgent(env, avoid_hazards=True)
        mdp_agent = ValueIterationAgent(env, gamma=0.95)
        rl_agent = QLearningAgent(env)

        agents = [csp_agent, mdp_agent, rl_agent]
        for agent in agents:
            t0 = time.perf_counter()
            agent.prepare(episodes=1500, seed=5_000 + idx*100)
            prep_time = time.perf_counter() - t0
            agent_metrics = evaluate_agent(env, agent, episodes=120, seed=10_000 + idx*100)
            method_name = type(agent).__name__.replace("Agent", "")
            rows.append(
                _row(
                    scenario=str(scenario["name"]),
                    size=instance.size,
                    method=method_name,
                    prep_time_sec=prep_time if prep_time > 1e-6 else 0.0,
                    metrics=agent_metrics,
                )
            )

    main_df = pd.DataFrame(rows)
    main_df.to_csv(RESULTS_DIR / "main_results.csv", index=False)
    return main_df


def run_scalability_experiments() -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []

    for size in scalability_sizes():
        instance: GridInstance = generate_instance(
            size=size,
            obstacle_density=0.18,
            hazard_density=0.10,
            slip_prob=0.15,
            step_limit=4 * size,
            seed=900 + size,
        )
        env = StochasticGridWorld(instance, seed=1)
        csp_agent = CSPReplanningAgent(env, avoid_hazards=True)
        mdp_agent = ValueIterationAgent(env, gamma=0.95)
        rl_agent = QLearningAgent(env)
        agents = [csp_agent, mdp_agent, rl_agent]
        for agent in agents:
            t0 = time.perf_counter()
            agent.prepare(episodes=1500, seed=70_000 + 10*size)
            prep_time = time.perf_counter() - t0
            agent_metrics = evaluate_agent(env, agent, episodes=80, seed=50_000 + 10*size)
            method_name = type(agent).__name__.replace("Agent", "")
            rows.append(
                _row(
                    scenario="scalability",
                    size=size,
                    method=method_name,
                    prep_time_sec=prep_time if prep_time > 1e-6 else 0.0,
                    metrics=agent_metrics,
                )
            )

    scalability_df = pd.DataFrame(rows)
    scalability_df.to_csv(RESULTS_DIR / "scalability_results.csv", index=False)
    return scalability_df


def save_plots(main_df: pd.DataFrame, scalability_df: pd.DataFrame) -> None:
    order = ["easy", "medium", "hard"]
    pivot_success = (
        main_df.pivot(index="scenario", columns="method", values="success_rate")
        .reindex(order)
        .fillna(0.0)
    )

    ax = pivot_success.plot(kind="bar", figsize=(8, 4.5))
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Scenario")
    ax.set_ylim(0, 1.05)
    ax.set_title("Success rate by method")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "main_success_rate.png", dpi=220)
    plt.close()

    pivot_return = (
        main_df.pivot(index="scenario", columns="method", values="avg_return")
        .reindex(order)
        .fillna(0.0)
    )
    ax = pivot_return.plot(kind="bar", figsize=(8, 4.5))
    ax.set_ylabel("Average return")
    ax.set_xlabel("Scenario")
    ax.set_title("Average return by method")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "main_avg_return.png", dpi=220)
    plt.close()

    runtime_df = scalability_df.copy()
    runtime_pivot = runtime_df.pivot(index="size", columns="method", values="total_time_sec")
    ax = runtime_pivot.plot(marker="o", figsize=(8, 4.5))
    ax.set_xlabel("Grid size (N for N x N)")
    ax.set_ylabel("Total time (seconds)")
    ax.set_title("Scalability runtime")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "scalability_runtime.png", dpi=220)
    plt.close()

    success_pivot = runtime_df.pivot(index="size", columns="method", values="success_rate")
    ax = success_pivot.plot(marker="o", figsize=(8, 4.5))
    ax.set_xlabel("Grid size (N for N x N)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Scalability success rate")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "scalability_success.png", dpi=220)
    plt.close()


def save_summary(main_df: pd.DataFrame, scalability_df: pd.DataFrame) -> None:
    rounded_main = main_df.copy()
    rounded_scal = scalability_df.copy()

    for col in [
        "success_rate",
        "avg_return",
        "avg_steps_on_success",
        "prep_time_sec",
        "eval_time_sec",
        "total_time_sec",
    ]:
        rounded_main[col] = rounded_main[col].map(lambda x: round(float(x), 4))
        rounded_scal[col] = rounded_scal[col].map(lambda x: round(float(x), 4))

    with (RESULTS_DIR / "result_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Main experiments\n")
        f.write(rounded_main.to_string(index=False))
        f.write("\n\nScalability experiments\n")
        f.write(rounded_scal.to_string(index=False))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    main_df = run_main_experiments()
    scalability_df = run_scalability_experiments()
    save_plots(main_df, scalability_df)
    save_summary(main_df, scalability_df)

    print("Saved:")
    print(RESULTS_DIR / "main_results.csv")
    print(RESULTS_DIR / "scalability_results.csv")
    print(RESULTS_DIR / "result_summary.txt")
    print(RESULTS_DIR / "main_success_rate.png")
    print(RESULTS_DIR / "main_avg_return.png")
    print(RESULTS_DIR / "scalability_runtime.png")
    print(RESULTS_DIR / "scalability_success.png")


if __name__ == "__main__":
    main()
