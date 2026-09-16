# Connectome Flight

Autonomous FPV quadrotor racing and navigation using a Drosophila connectome as a recurrent high-level flight policy.

## Architecture

```text
FPV camera -> visual encoder --------┐
IMU / velocity / navigation --------┼-> Drosophila connectome -> velocity / yaw targets
Gate geometry / mission ------------┘                              |
                                                                   v
                                                        conventional flight controller
                                                        stabilization / mixer / safety
                                                                   |
                                                                   v
                                                             ESCs -> motors
```

The connectome is **not** the motor-rate controller. The conventional flight controller remains responsible for high-rate attitude/rate stabilization, motor mixing, battery protection, command timeout and failsafe behavior.

## Development stages

1. State-based racing in simulation.
2. Procedural gates and curriculum learning.
3. Dynamics and sensor randomization.
4. Visual feature inputs.
5. Camera encoder training.
6. Hardware-in-the-loop.
7. Embedded optimization and conservative real-flight validation.

The first model keeps the full selected Haltere connectome intact. Pruning, quantization and other compression are later optimization stages after measured baselines exist.

## Upstream

The neural core is based on [Haltere](https://github.com/skulitom/haltere). Haltere provides a connectome-constrained recurrent network with fixed connectome topology/sign structure and trainable neural dynamics, edge gains and sensory encoders. The drone-specific code in this repository is kept behind adapters so simulator and hardware interfaces can evolve independently.

Flightmare is the planned high-fidelity quadrotor simulator because its physics and rendering systems are decoupled and it exposes sensors and parallel RL interfaces.

## Repository layout

```text
connectome-flight/
  configs/                  experiment configuration
  docs/                     architecture and experiment notes
  src/connectome_flight/
    brain/                  Haltere adapter and policy interfaces
    envs/                   lightweight racing environment and Flightmare adapter
    control/                high-level command contract
    training/               rollout and training utilities
    evaluation/             metrics and benchmarks
    perception/             visual encoder interfaces
  tests/                    unit and smoke tests
```

## Safety boundary

Never connect an experimental learned policy directly to motor PWM. The intended command path is learned high-level target -> independent flight controller -> ESC/motor output, with independent failsafes.

## Status

**Phase 0 / foundation:** repository architecture and contracts are being implemented. Simulation integration comes before neural training; no claim of autonomous flight is made until the corresponding simulator and hardware tests exist.
