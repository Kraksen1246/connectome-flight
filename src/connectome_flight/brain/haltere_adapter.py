from __future__ import annotations

"""Thin adapter boundary for the upstream Haltere ConnectomeRNN.

No Haltere internals are duplicated here. The adapter is intentionally dependency-injected so
training code can be tested without downloading connectome data.
"""

import numpy as np

from .policy import PolicyState, RecurrentPolicy


class HalterePolicy(RecurrentPolicy):
    """Wrap a constructed Haltere ``ConnectomeRNN``.

    ``brain`` is expected to expose ``init_state(B)`` and ``forward(obs, state)``. Observation
    preprocessing is supplied by ``observation_builder`` so the mapping from drone state to
    Haltere populations remains an explicit experiment configuration rather than hidden logic.
    """

    def __init__(self, brain, observation_builder, device: str = "cpu"):
        self.brain = brain
        self.observation_builder = observation_builder
        self.device = device

    def initial_state(self, batch_size: int = 1) -> PolicyState:
        return PolicyState(self.brain.init_state(batch_size))

    def step(self, observation: np.ndarray, state: PolicyState):
        import torch

        obs = self.observation_builder(observation)
        obs = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device).reshape(1, -1) for k, v in obs.items()}
        with torch.no_grad():
            action, next_state, _ = self.brain.forward(obs, state.value)
        return action[0].detach().cpu().numpy().astype(np.float32), PolicyState(next_state)
