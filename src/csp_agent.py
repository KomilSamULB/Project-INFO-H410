from __future__ import annotations

from collections import deque
from typing import Optional

from .agent import Agent
from .environment import ACTIONS, State, StochasticGridWorld


def _manhattan(a: State, b: State) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class CSPReplanningAgent(Agent):
    def __init__(self, env: StochasticGridWorld, avoid_hazards: bool = True) -> None:
        self.env = env
        self.avoid_hazards = avoid_hazards
        self.first_print = False

    def _planning_blocked(self, hazards_blocked: bool = True) -> set[State]:
        blocked = set(self.env.instance.obstacles)
        if self.avoid_hazards and hazards_blocked:
            blocked.update(self.env.instance.hazards)
        return blocked

    def _ordered_neighbors(
        self,
        state: State,
        goal: State,
        blocked: set[State],
    ) -> list[State]:
        candidates: list[State] = []
        for action in ACTIONS:
            nxt = self.env.deterministic_next_state(state, action)
            if nxt == state or nxt in blocked:
                continue
            candidates.append(nxt)
        candidates.sort(key=lambda cell: _manhattan(cell, goal))
        return candidates

    def _shortest_feasible_path(
        self,
        start: State,
        goal: State,
        blocked: set[State],
        hazards_blocked: bool = True
    ) -> Optional[list[State]]:
        if start == goal:
            return [start]

        # Dequeue keeps track of the nodes to explore, starting with the initial state.
        queue: deque[State] = deque([start])

        # Parents keeps track of the parents of the node
        # For ex parent[A] = B means that we reached A from B. 
        parent: dict[State, Optional[State]] = {start: None}

        # We implement a breadth-first-search to find the shortst path
        # from start to goal, while avoiding blocked cells.
        while queue:
            current = queue.popleft()
            for nxt in self._ordered_neighbors(current, goal, blocked):
                if nxt in parent:
                    continue
                # We keep track of the path we followed
                parent[nxt] = current
                if nxt == goal:
                    # When we arrive at the goal, we can reconstruct the path by following the parents
                    # And because it's a breadth-first-search, we are guaranteed that this path is the shortest one.
                    path = [goal]
                    cursor: Optional[State] = goal
                    while cursor is not None:
                        cursor = parent[cursor]
                        if cursor is not None:
                            path.append(cursor)
                    path.reverse()
                    return path
                # We add the next state to the queue to explore it later
                queue.append(nxt)

        if (hazards_blocked):
            # If we didn't find a path while blocking hazards, we can try again without blocking them
            return self._shortest_feasible_path(
                start, 
                goal, 
                self._planning_blocked(hazards_blocked=False), 
                hazards_blocked=False
            )
        return None

    def plan_path(self, start: State, remaining_steps: int) -> Optional[list[State]]:
        if remaining_steps <= 0:
            print("No remaining steps")
            return None

        goal = self.env.instance.goal
        blocked = self._planning_blocked(hazards_blocked=True)

        path = self._shortest_feasible_path(start, goal, blocked)
        if path is None:
            return None

        return path

    def act(self, state: State, step_count: int) -> Optional[str]:
        if state == self.env.instance.goal:
            return "U"

        remaining = max(1, self.env.instance.step_limit - step_count)
        path = self.plan_path(state, remaining_steps=remaining)

        if path is not None and len(path) >= 2:
            # print("Used normal went to " + str(path[1]) + " wich is " + str(self.env.is_blocked(path[1])))
            return self.env.action_from_to(path[0], path[1])

        return None
    
    def prepare(self, episodes: int = 0, seed: int = 0) -> None:
        return
