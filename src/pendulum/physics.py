import jax
import jax.numpy as jnp

DEFAULT_PARAMS = {"g": 9.81, "l": 1.0, "mc": 1.0, "mp": 0.1}


def dynamics(state, force, env_params):
    x, theta, x_dot, theta_dot = state
    g = env_params["g"]
    pole_len = env_params["l"]
    mc = env_params["mc"]
    mp = env_params["mp"]

    sin_t = jnp.sin(theta)
    cos_t = jnp.cos(theta)
    denom = mc + mp * sin_t**2

    x_ddot = (mp * pole_len * theta_dot**2 * sin_t - mp * g * sin_t * cos_t + force) / denom
    theta_ddot = ((mc + mp) * g * sin_t - mp * pole_len * theta_dot**2 * sin_t * cos_t - force * cos_t) / (
        pole_len * denom
    )

    return jnp.array([x_dot, theta_dot, x_ddot, theta_ddot])


def step(state, torque, dt, env_params):
    k1 = dynamics(state, torque, env_params)
    k2 = dynamics(state + 0.5 * dt * k1, torque, env_params)
    k3 = dynamics(state + 0.5 * dt * k2, torque, env_params)
    k4 = dynamics(state + dt * k3, torque, env_params)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(state0, controls, dt, env_params):
    def scan_fn(state, torque):
        next_state = step(state, torque, dt, env_params)
        return next_state, next_state

    _, trajectory = jax.lax.scan(scan_fn, state0, controls)
    return trajectory
