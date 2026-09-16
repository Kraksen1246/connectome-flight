from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Gate:
    center: np.ndarray
    radius: float = 1.0


@dataclass
class RacingState:
    position: np.ndarray
    velocity: np.ndarray
    yaw: float
    next_gate: int


class SimpleRacingEnv:
    """Dependency-free smoke environment for the policy contract.

    This is intentionally not a replacement for Flightmare. It provides a deterministic,
    differentiable-policy-friendly API for unit tests before the simulator is installed.
    """

    def __init__(self, gates: list[Gate] | None = None, dt: float = 0.02, max_speed: float = 12.0):
        self.dt = float(dt)
        self.max_speed = float(max_speed)
        self.gates = gates or [
            Gate(np.array([10.0, 0.0, 1.5], dtype=np.float32)),
            Gate(np.array([20.0, 3.0, 1.5], dtype=np.float32)),
            Gate(np.array([30.0, 0.0, 1.5], dtype=np.float32)),
        ]
        self.state = RacingState(np.zeros(3, np.float32), np.zeros(3, np.float32), 0.0, 0)
        self.steps = 0

    def reset(self) -> np.ndarray:
        self.state = RacingState(np.zeros(3, np.float32), np.zeros(3, np.float32), 0.0, 0)
        self.steps = 0
        return self.observation()

    def observation(self) -> np.ndarray:
        if self.state.next_gate >= len(self.gates):
            rel = np.zeros(3, dtype=np.float32)
        else:
            rel = self.gates[self.state.next_gate].center - self.state.position
        return np.concatenate([rel, self.state.velocity, [np.sin(self.state.yaw), np.cos(self.state.yaw)]])

    def step(self, command: np.ndarray):
        command = np.asarray(command, dtype=np.float32).reshape(4)
        desired = np.clip(command[:3], -self.max_speed, self.max_speed)
        self.state.velocity += (desired - self.state.velocity) * min(1.0, 4.0 * self.dt)
        self.state.velocity = np.clip(self.state.velocity, -self.max_speed, self.max_speed)
        self.state.position += self.state.velocity * self.dt
        self.state.yaw += float(command[3]) * self.dt
        self.steps += 1

        reward = float(np.dot(self.state.velocity, self._gate_direction())) * self.dt
        terminated = False
        if self.state.next_gate < len(self.gates):
            gate = self.gates[self.state.next_gate]
            distance = float(np.linalg.norm(self.state.position - gate.center))
            if distance <= gate.radius:
                self.state.next_gate += 1
                reward += 10.0
                if self.state.next_gate == len(self.gates):
                    terminated = True
        truncated = self.steps >= 2500
        return self.observation(), reward, terminated, truncated, {"gate": self.state.next_gate}

    def _gate_direction(self) -> np.ndarray:
        if self.state.next_gate >= len(self.gates):
            return np.zeros(3, np.float32)
        d = self.gates[self.state.next_gate].center - self.state.position
        n = np.linalg.norm(d)
        return d / max(n, 1e-6)
