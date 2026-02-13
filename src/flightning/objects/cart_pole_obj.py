import jax
import jax_dataclasses as jdc
import numpy as np
from jax import numpy as jnp
from flightning.utils.pytrees import field_jnp, CustomPyTree


@jdc.pytree_dataclass
class CartPoleState(CustomPyTree):
    x: jax.Array = field_jnp(0.0)
    theta: jax.Array = field_jnp(0.0)
    x_dot: jax.Array = field_jnp(0.0)
    theta_dot: jax.Array = field_jnp(0.0)

    def detached(self): # detached -> stop gradient flowing through it.
        """Return a detached copy with no gradients."""
        return CartPoleState(
            x=jax.lax.stop_gradient(self.x),
            theta=jax.lax.stop_gradient(self.theta),
            x_dot=jax.lax.stop_gradient(self.x_dot),
            theta_dot=jax.lax.stop_gradient(self.theta_dot),
        )

    def as_vector(self):
        """Convert state to a flat vector."""
        return jnp.array([self.x, self.theta, self.x_dot, self.theta_dot])

    @classmethod
    def from_vector(cls, vector):
        """Create state from a flat vector."""
        return cls(
            x=vector[0],
            theta=vector[1],
            x_dot=vector[2],
            theta_dot=vector[3],
        )


class CartPole:
    """Cart-pole dynamics

    State: [x, theta, x_dot, theta_dot]
        - x: cart position
        - theta: pole angle from vertical (0 = upright)
        - x_dot: cart velocity
        - theta_dot: pole angular velocity

    Action: force applied to the cart [N]
    """

    def __init__(
        self,
        *,
        gravity=9.81,  # [m/s^2]
        length=1.0,  # [m] pole length (to center of mass)
        mass_cart=1.0,  # [kg] cart mass
        mass_pole=0.1,  # [kg] pole mass
        friction_cart=0.0,  # [N/(m/s)] cart friction coefficient
        friction_pole=0.0,  # [N*m/(rad/s)] pole friction coefficient
    ):
        self._gravity = gravity
        self._length = length
        self._mass_cart = mass_cart
        self._mass_pole = mass_pole
        self._friction_cart = friction_cart
        self._friction_pole = friction_pole

        # Total mass
        self._mass_total = mass_cart + mass_pole

        # Pole inertia around pivot (point mass approximation)
        self._inertia = mass_pole * length ** 2

    def default_state(self):
        return CartPoleState(x=0.0, theta=0.0, x_dot=0.0, theta_dot=0.0)

    def create_state(
        self,
        x: float = 0.0,
        theta: float = 0.0,
        x_dot: float = 0.0,
        theta_dot: float = 0.0,
    ):
        """Create a cart-pole state with specified values."""
        return CartPoleState(
            x=jnp.array(x),
            theta=jnp.array(theta),
            x_dot=jnp.array(x_dot),
            theta_dot=jnp.array(theta_dot),
        )

    def step(
        self,
        state: CartPoleState,
        force: jax.Array,
        dt: jax.Array,
    ) -> CartPoleState:
        """Simulate one time step.

        Args:
            state: Current cart-pole state
            force: Force applied to cart [N]
            dt: Time step [s]

        Returns:
            Next cart-pole state after dt seconds
        """
        # Use RK4 integration for accuracy
        def dynamics(s: CartPoleState, f: jax.Array):
            """Compute state derivatives."""
            # Extract state
            theta = s.theta
            x_dot = s.x_dot
            theta_dot = s.theta_dot

            # Precompute trig functions
            sin_theta = jnp.sin(theta)
            cos_theta = jnp.cos(theta)

            # Compute accelerations from equations of motion
            # These are derived from Lagrangian mechanics

            # Effective force on cart
            f_eff = f - self._friction_cart * x_dot

            # Torque on pole
            tau_eff = -self._friction_pole * theta_dot

            # Denominator for both accelerations
            denom = self._mass_total - self._mass_pole * cos_theta ** 2

            # Cart acceleration
            x_ddot = (
                f_eff
                + self._mass_pole * sin_theta * (
                    self._length * theta_dot ** 2
                    + self._gravity * cos_theta
                )
            ) / denom

            # Pole angular acceleration
            theta_ddot = (
                -f_eff * cos_theta
                - self._mass_pole * self._length * theta_dot ** 2 * cos_theta * sin_theta
                - self._mass_total * self._gravity * sin_theta
                + tau_eff / self._length
            ) / (self._length * denom)

            # Return state derivative
            return CartPoleState(
                x=x_dot,
                theta=theta_dot,
                x_dot=x_ddot,
                theta_dot=theta_ddot,
            )

        # RK4 integration
        k1 = dynamics(state, force)

        state_k2 = CartPoleState(
            x=state.x + 0.5 * dt * k1.x,
            theta=state.theta + 0.5 * dt * k1.theta,
            x_dot=state.x_dot + 0.5 * dt * k1.x_dot,
            theta_dot=state.theta_dot + 0.5 * dt * k1.theta_dot,
        )
        k2 = dynamics(state_k2, force)

        state_k3 = CartPoleState(
            x=state.x + 0.5 * dt * k2.x,
            theta=state.theta + 0.5 * dt * k2.theta,
            x_dot=state.x_dot + 0.5 * dt * k2.x_dot,
            theta_dot=state.theta_dot + 0.5 * dt * k2.theta_dot,
        )
        k3 = dynamics(state_k3, force)

        state_k4 = CartPoleState(
            x=state.x + dt * k3.x,
            theta=state.theta + dt * k3.theta,
            x_dot=state.x_dot + dt * k3.x_dot,
            theta_dot=state.theta_dot + dt * k3.theta_dot,
        )
        k4 = dynamics(state_k4, force)

        # Combine RK4 steps
        next_state = CartPoleState(
            x=state.x + (dt / 6.0) * (k1.x + 2 * k2.x + 2 * k3.x + k4.x),
            theta=state.theta + (dt / 6.0) * (k1.theta + 2 * k2.theta + 2 * k3.theta + k4.theta),
            x_dot=state.x_dot + (dt / 6.0) * (k1.x_dot + 2 * k2.x_dot + 2 * k3.x_dot + k4.x_dot),
            theta_dot=state.theta_dot + (dt / 6.0) * (k1.theta_dot + 2 * k2.theta_dot + 2 * k3.theta_dot + k4.theta_dot),
        )

        return next_state


if __name__ == "__main__":
    # Test the cart-pole model
    cart_pole = CartPole(mass_cart=1.0, mass_pole=0.1, length=0.5)
    state = cart_pole.create_state(x=0.0, theta=0.1, x_dot=0.0, theta_dot=0.0)

    # Simulate for 1 second with no force
    force = jnp.array(0.0)
    dt = 0.02

    print("Initial state:", state.as_vector())
    for i in range(50):
        state = cart_pole.step(state, force, dt)
    print("Final state after 1s:", state.as_vector())