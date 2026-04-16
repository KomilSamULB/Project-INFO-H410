from __future__ import annotations

from typing import Optional, Protocol

from .environment import Action, State

class Agent(Protocol):
	"""Common interface implemented by all decision agents."""

	def prepare(self, episodes: int = 0, seed: int = 0) -> None:
		"""Optional pre-computation stage before evaluation."""
		"""Episodes and seed are used to train the agent so they are optional."""

	def act(self, state: State, step_count: int) -> Optional[Action]:
		"""Return an action for the current state, or None to give up."""