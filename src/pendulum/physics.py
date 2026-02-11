import jax
import jax.numpy as jnp

DEFAULT_PARAMS = {"g": 9.81, "l": 1.0, "m": 1.0}


def dynamics(state, torque, env_params):
    angle, angular_v = state

    g_impac = (env_params["g"] / env_params["l"]) * jnp.sin(angle)
    manual_impac = torque / (env_params["m"] * (env_params["l"] ** 2))
    angular_acc = g_impac + manual_impac

    return jnp.array([angular_v, angular_acc])


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
