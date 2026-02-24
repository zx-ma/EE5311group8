from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from pendulum.physics import step


class Policy(eqx.Module):
    layers: list

    def __init__(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(5, 512, key=k1),
            eqx.nn.Linear(512, 512, key=k2),
            eqx.nn.Linear(512, 1, key=k3),
        ]

    def __call__(self, obs):
        x = jax.nn.relu(self.layers[0](obs))
        x = jax.nn.relu(self.layers[1](x))
        return self.layers[2](x).squeeze()


def state_to_obs(state):
    x, theta, x_dot, theta_dot = state
    return jnp.array([x, jnp.sin(theta), jnp.cos(theta), x_dot, theta_dot])


def simulate_nn(policy, state0, dt, n_steps, env_params):
    def scan_fn(state, _):
        obs = state_to_obs(state)
        force = jnp.clip(policy(obs), -200.0, 200.0)
        next_state = step(state, force, dt, env_params)
        return next_state, (next_state, force)

    _, (traj, forces) = jax.lax.scan(scan_fn, state0, jnp.arange(n_steps))
    return traj, forces


def nn_loss(policy, state0, dt, n_steps, env_params, lam=0.001):
    traj, forces = simulate_nn(policy, state0, dt, n_steps, env_params)
    # (1 - cos(theta))^2 penalizes both upright deviation and hanging down
    theta_cost = jnp.mean((1 - jnp.cos(traj[:, 1])) ** 2)
    x_cost = jnp.mean(traj[:, 0] ** 2)
    vel_cost = jnp.mean(traj[:, 3] ** 2)  # penalize angular velocity for stability
    force_cost = jnp.mean(forces**2)
    return theta_cost + 0.1 * x_cost + 0.1 * vel_cost + lam * force_cost


@eqx.filter_jit
def train_step(policy, opt_state, state0, dt, n_steps, env_params, optimizer):
    loss, grads = eqx.filter_value_and_grad(nn_loss)(policy, state0, dt, n_steps, env_params)
    updates, new_opt_state = optimizer.update(grads, opt_state, eqx.filter(policy, eqx.is_array))
    new_policy = eqx.apply_updates(policy, updates)
    return loss, new_policy, new_opt_state


def save_policy(policy, path="data/neural/policy.eqx"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, policy)
    print(f"  saved policy to {path}")


def load_policy(path="data/neural/policy.eqx", seed=0):
    skeleton = Policy(jax.random.PRNGKey(seed))
    policy = eqx.tree_deserialise_leaves(path, skeleton)
    print(f"  loaded policy from {path}")
    return policy


def _train_single(state0, dt, n_steps, env_params, lr, epochs, seed):
    key = jax.random.PRNGKey(seed)
    policy = Policy(key)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    for i in range(epochs):
        loss, policy, opt_state = train_step(policy, opt_state, state0, dt, n_steps, env_params, optimizer)
        if i % 50 == 0 or i == epochs - 1:
            print(f"  epoch {i:4d} | loss {float(loss):.6f}")

    final_loss = float(nn_loss(policy, state0, dt, n_steps, env_params))
    return policy, final_loss


def train_nn(state0, dt, n_steps, env_params, lr=0.001, epochs=500, n_seeds=3):
    best_policy, best_loss = None, float("inf")
    for seed in range(n_seeds):
        print(f"\n  === seed {seed} ===")
        policy, loss = _train_single(state0, dt, n_steps, env_params, lr, epochs, seed)
        if loss < best_loss:
            best_policy, best_loss = policy, loss
        print(f"  seed {seed} final loss: {loss:.6f}")
    print(f"\n  best loss: {best_loss:.6f}")
    return best_policy
