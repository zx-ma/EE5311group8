import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_trajectory(traj, dt):
    t = jnp.arange(traj.shape[0]) * dt
    labels = ["x (m)", "theta (rad)", "x_dot (m/s)", "theta_dot (rad/s)"]

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, traj[:, i])
        ax.set_ylabel(label)
    axes[-1].set_xlabel("time (s)")

    fig.tight_layout()
    return fig


def animate_cartpole(traj, dt, pole_len=1.0, skip=5, save_path=None):
    cart_xs = traj[:, 0]
    angles = traj[:, 1]

    x_range = float(jnp.abs(cart_xs).max()) + pole_len + 0.5
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(-x_range, x_range)
    ax.set_ylim(-pole_len * 1.5, pole_len * 1.5)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.5)

    cart_w, cart_h = 0.3, 0.15
    cart_patch = plt.Rectangle((0, 0), cart_w, cart_h, fc="steelblue")
    ax.add_patch(cart_patch)
    (rod,) = ax.plot([], [], "o-", color="royalblue", lw=2, markersize=8)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def update(frame):
        i = frame * skip
        cx = float(cart_xs[i])
        theta = float(angles[i])
        cart_patch.set_xy((cx - cart_w / 2, -cart_h / 2))
        pole_x = cx + pole_len * jnp.sin(theta)
        pole_y = pole_len * jnp.cos(theta)
        rod.set_data([cx, pole_x], [0, pole_y])
        time_text.set_text(f"t = {i * dt:.2f}s")
        return cart_patch, rod, time_text

    n_frames = len(angles) // skip
    anim = FuncAnimation(fig, update, frames=n_frames, interval=dt * skip * 1000, blit=True, repeat=False)
    if save_path:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer="pillow")
    plt.show()
    return anim
