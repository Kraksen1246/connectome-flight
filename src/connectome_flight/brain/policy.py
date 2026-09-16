from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class PolicyState:
    """Opaque recurrent state owned by a policy implementation."""

    value: object | None = None


class RecurrentPolicy(ABC):
    """Stable interface shared by Haltere and baseline policies."""

    @abstractmethod
    def initial_state(self, batch_size: int = 1) -> PolicyState:
        raise NotImplementedError

    @abstractmethod
    def step(self, observation: np.ndarray, state: PolicyState) -> tuple[np.ndarray, PolicyState]:
        """Return normalized high-level action [vx, vy, vz, yaw_rate]."""
        raise NotImplementedError


class ZeroPolicy(RecurrentPolicy):
    """Safe deterministic baseline used by smoke tests."""

    def initial_state(self, batch_size: int = 1) -> PolicyState:
        return PolicyState(None)

    def step(self, observation: np.ndarray, state: PolicyState):
        del observation
        return np.zeros(4, dtype=np.float32), state
