from __future__ import annotations

"""Flightmare integration boundary.

The concrete Flightmare API is intentionally isolated here. Do not import Flightmare throughout the
brain or training packages. The adapter should translate Flightmare state/sensors into the canonical
Connectome Flight observation contract and translate FlightCommand into the conventional controller
interface.
"""


class FlightmareNotInstalled(RuntimeError):
    pass


class FlightmareAdapter:
    def __init__(self, *args, **kwargs):
        try:
            import flightgym  # noqa: F401
        except ImportError as exc:
            raise FlightmareNotInstalled(
                "Flightmare is not installed. Install/configure Flightmare separately before using this adapter."
            ) from exc
        self.args = args
        self.kwargs = kwargs

    def reset(self):
        raise NotImplementedError("Bind this method to the selected Flightmare/Gymnasium environment.")

    def step(self, command):
        raise NotImplementedError("Bind this method to the selected Flightmare/Gymnasium environment.")
