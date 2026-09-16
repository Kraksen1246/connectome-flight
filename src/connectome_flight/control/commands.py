from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FlightCommand:
    """High-level command handed to an independent flight controller.

    Units are SI. The learned policy does not command individual motors.
    """

    vx: float
    vy: float
    vz: float
    yaw_rate: float

    def as_array(self) -> np.ndarray:
        return np.asarray([self.vx, self.vy, self.vz, self.yaw_rate], dtype=np.float32)

    @classmethod
    def clipped(cls, value: np.ndarray, limits: np.ndarray) -> "FlightCommand":
        x = np.asarray(value, dtype=np.float32).reshape(4)
        y = np.clip(x, -np.asarray(limits, dtype=np.float32), np.asarray(limits, dtype=np.float32))
        return cls(*map(float, y))
