import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_trajectory(traj, dt):
    t = jnp.arange(traj.shape[0]) * dt

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(t, traj[:, 0])
    ax1.set_ylabel("angle (rad)")

    ax2.plot(t, traj[:, 1])
    ax2.set_ylabel("angular velocity (rad/s)")
    ax2.set_xlabel("time (s)")

    fig.tight_layout()
    return fig


def animate_pendulum(traj, dt, pendu_len=1.0, skip=5):
    angles = traj[:, 0]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-pendu_len * 1.3, pendu_len * 1.3)
    ax.set_ylim(-pendu_len * 1.3, pendu_len * 1.3)
    ax.set_aspect("equal")

    (rod,) = ax.plot([], [], "o-", color="steelblue", lw=2, markersize=8)
    ax.plot(0, 0, "ks", markersize=6)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def update(frame):
        i = frame * skip
        theta = float(angles[i])
        x = pendu_len * jnp.sin(theta)
        y = pendu_len * jnp.cos(theta)
        rod.set_data([0, x], [0, y])
        time_text.set_text(f"t = {i * dt:.2f}s")
        return rod, time_text

    n_frames = len(angles) // skip
    anim = FuncAnimation(fig, update, frames=n_frames, interval=dt * skip * 1000, blit=True)
    plt.show()
    return anim
