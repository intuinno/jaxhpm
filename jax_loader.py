import jax.numpy as jnp
from jax import random, jit
import mnist
from functools import partial
import gymnasium as gym


class JaxMNISTLoader:

    def __init__(self):
        self.width, self.height = 64, 64
        mnist_width, mnist_height = 28, 28
        self.lims = (x_lim, y_lim) = (
            self.width - mnist_width,
            self.height - mnist_height,
        )

    @partial(jit, static_argnums=(0,))
    def step(self, env_state, action):
        state, key = env_state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag * (2 * action - 1)
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        done = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta > self.theta_threshold_radians)
            | (theta < -self.theta_threshold_radians)
        )

        reward = 1.0

        env_state = jnp.array([x, x_dot, theta, theta_dot]), key
        env_state = self._maybe_reset(env_state, done)
        new_state = env_state[0]
        return env_state, self._get_obsv(new_state), reward, done, {}

    def _get_obsv(self, state):
        return state

    def _maybe_reset(self, env_state, done):
        key = env_state[1]
        return jax.lax.cond(done, self._reset, lambda key: env_state, key)

    def _reset(self, key):
        new_state = random.uniform(
            key, minval=-self.random_limit, maxval=self.random_limit, shape=(4,)
        )
        new_key = random.split(key)[0]
        return new_state, new_key

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        env_state = self._reset(key)
        initial_state = env_state[0]
        return env_state, self._get_obsv(initial_state)


seed = 0
key = random.PRNGKey(seed)
env = JaxCartPole()
env_state, inital_obsv = env.reset(key)
action = 1
env_state, obsv, reward, done, info = env.step(env_state, 1)
