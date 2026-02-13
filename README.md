# EE5311 Group8

using `uv` for python env

## start
```sh
uv sync  # if ur computer only has integrate gpu or..
uv sync --extra cuda # if u have nvidia card
```
do `uv add <package name>` to install package u need



## files

- `src/physics.py` - pendulum ODE + RK4 integrator, all in jax so its differentiable
  - todo: explain rk4 https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods? maybe, feel free to remove it and use other methods
- `src/visualize.py` - plotting things?


## test 
(or just source the venv and call python)
- uv run tests/test_physics.py 

# seems good? 
- uv run tests/visualize.py 
- uv run tests/test_pid.py

# external 
- https://en.wikipedia.org/wiki/Inverted_pendulum
