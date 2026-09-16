import numpy as np

from connectome_flight.brain.policy import ZeroPolicy
from connectome_flight.control import FlightCommand
from connectome_flight.envs import SimpleRacingEnv


def test_command_shape():
    command = FlightCommand(1, 2, 3, 4)
    assert command.as_array().shape == (4,)
    assert command.as_array().dtype == np.float32


def test_smoke_environment_completes_steps():
    env = SimpleRacingEnv()
    obs = env.reset()
    policy = ZeroPolicy()
    state = policy.initial_state()
    assert obs.shape == (8,)
    for _ in range(3):
        action, state = policy.step(obs, state)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        assert "gate" in info
        assert not terminated
        assert not truncated


def test_environment_moves_toward_gate_with_command():
    env = SimpleRacingEnv()
    env.reset()
    before = np.linalg.norm(env.gates[0].center - env.state.position)
    for _ in range(20):
        env.step(np.array([8, 0, 0, 0], dtype=np.float32))
    after = np.linalg.norm(env.gates[0].center - env.state.position)
    assert after < before
