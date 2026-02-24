import functools

import jax
import jax.numpy as jnp
import optax

from controllers.pid import simulate_pid


def pid_loss(pid_params, state0, dt, n_steps, env_params, lam=0.001):
    traj, forces = simulate_pid(state0, pid_params, dt, n_steps, env_params)
    theta_cost = jnp.mean(traj[:, 1] ** 2)
    x_cost = jnp.mean(traj[:, 0] ** 2)
    force_cost = jnp.mean(forces**2)
    return theta_cost + x_cost + lam * force_cost


@functools.partial(jax.jit, static_argnums=(3,))
def train_step(params, opt_state, state0, n_steps, dt, env_params, optimizer):
    loss, grads = jax.value_and_grad(pid_loss)(params, state0, dt, n_steps, env_params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state


def optimize_pid(init_params, state0, dt, n_steps, env_params, lr=0.001, epochs=300):
    optimizer = optax.adamw(lr)
    opt_state = optimizer.init(init_params)
    params = init_params

    # close over optimizer since jit cant trace it
    @functools.partial(jax.jit, static_argnums=(3,))
    def step(params, opt_state, state0, n_steps, dt, env_params):
        loss, grads = jax.value_and_grad(pid_loss)(params, state0, dt, n_steps, env_params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state

    for i in range(epochs):
        loss, params, opt_state = step(params, opt_state, state0, n_steps, dt, env_params)
        if i % 50 == 0 or i == epochs - 1:
            print(f"epoch {i:4d} | loss {float(loss):.6f} | params {params}")

    return params
