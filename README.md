# EE5311 Group8

using `uv` for python env

## start
```sh
uv sync  # if ur computer only has integrate gpu or..
uv sync --extra cuda # if u have nvidia card
```
do `uv add <package name>` to install package u need

## project structure

```
src/
  pendulum/
    physics.py    - cart-pole ODE + RK4 integrator, all in jax so its differentiable
    visualize.py  - trajectory plots + cart-pole animation (gif)
  controllers/
    pid.py              - PID controller (manual tuning)
    optimize_pid.py     - PID param optimization via adam (autodiff through physics)
    optimize_pid_lbfgs.py - PID param optimization via l-bfgs (second order)
    neural.py           - NN policy (MLP 5→512→512→1, equinox), trained via BPTT through differentiable sim
```

## run

```sh
# compare manual pid vs adam-optimized pid
uv run tests/test_compare_optimizers.py

# neural network controller (train from scratch, saves to data/neural/policy.eqx)
uv run tests/test_neural.py

# load saved model (skip training)
uv run tests/test_neural.py --load
```

## method overview

1. **differentiable physics** - cart-pole dynamics written in jax, so jax.grad can backprop through the simulator
2. **PID baseline** - manual params → optimize with adam (1st order) and l-bfgs (2nd order) via autodiff
3. **neural network** - MLP policy trained end-to-end through physics sim (BPTT), gradient clipping, multi-seed for robustness
4. loss = (1-cos θ)² + 0.1·x² + 0.1·θ̇² + 0.001·force² (penalize tip height deficit, position, velocity, energy)

## external
- https://en.wikipedia.org/wiki/Inverted_pendulum
