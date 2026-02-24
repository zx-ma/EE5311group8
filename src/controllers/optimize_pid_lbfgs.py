import jax
import optax

from controllers.optimize_pid import pid_loss


def optimize_pid_lbfgs(init_params, state0, dt, n_steps, env_params, maxiter=50):
    def loss_fn(params):
        return pid_loss(params, state0, dt, n_steps, env_params)

    solver = optax.lbfgs()
    state = solver.init(init_params)
    params = init_params
    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    for i in range(maxiter):
        loss, grads = value_and_grad_fn(params)
        updates, state = solver.update(grads, state, params, value=loss, grad=grads, value_fn=loss_fn)
        params = optax.apply_updates(params, updates)
        if i % 5 == 0 or i == maxiter - 1:
            print(f"  iter {i:3d} | loss {float(loss):.6f} | params {params}")

    return params
