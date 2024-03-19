import jax.numpy as jnp
import jax
import numpy as np
from jax import random, jit
import mnist
from functools import partial
import gymnasium as gym
import os
import imageio
import matplotlib.image


class JaxMNISTLoader:

    def __init__(self, mnist_images):
        self.width, self.height = 64, 64
        self.mnist_width, self.mnist_height = 28, 28
        self.lims = (x_lim, y_lim) = (
            self.width - self.mnist_width,
            self.height - self.mnist_height,
        )
        self.nums_per_image = 2
        self.images = jnp.array(mnist_images)
        self.num_images = len(mnist_images)

    @partial(jit, static_argnums=(0,))
    def step(self, env_state, action):
        state, key = env_state
        mnist_images, positions, velocs = state
        next_pos = positions + velocs

        velocs = jax.lax.select(
            ((next_pos < -2) | (next_pos > self.lims[0] + 2)), -1.0 * velocs, velocs
        )

        positions = positions + velocs
        next_state = (mnist_images, positions, velocs)
        next_env_state = next_state, key
        reward = 1.0
        done = False
        return next_env_state, self._get_obsv(next_state), reward, done, {}

    def _build_canvas(self, index, position):
        image = self.images[index]
        x, y = position.astype(int)
        x = jnp.where(x < 0, 0, x)
        x = jnp.where(x > self.lims[0], self.lims[0], x)
        y = jnp.where(y < 0, 0, y)
        y = jnp.where(y > self.lims[1], self.lims[1], y)
        # canvas = jnp.pad(
        #     image,
        #     (
        #         (x, self.width - x - self.mnist_width),
        #         (y, self.height - y - self.mnist_height),
        #     ),
        # )
        canvas = jnp.zeros((self.width, self.height), dtype="uint8")
        fig = jax.lax.dynamic_update_slice(canvas, image, (x, y))

        return fig

    def _get_obsv(self, state):
        indexes, positions, _ = state
        canvas = jnp.zeros((self.width, self.height))
        for i, p in zip(indexes, positions):
            canvas += self._build_canvas(i, p)
        canvas = jnp.where(canvas > 255.0, 255.0, canvas)
        return canvas

    def _maybe_reset(self, env_state, done):
        key = env_state[1]
        return jax.lax.cond(done, self._reset, lambda key: env_state, key)

    def _reset(self, key):
        new_key, subkey = random.split(key)
        direcs = jnp.pi * (random.uniform(subkey, shape=(self.nums_per_image,)) * 2 - 1)

        new_key, subkey = random.split(new_key)
        speeds = random.randint(subkey, (self.nums_per_image,), 0, 5) + 2
        velocs = jnp.array(
            [
                (speed * jnp.cos(direc), speed * jnp.sin(direc))
                for direc, speed in zip(direcs, speeds)
            ]
        )

        new_key, subkey = random.split(new_key)
        indexes = random.randint(subkey, (self.nums_per_image,), 0, self.num_images)

        new_key, subkey = random.split(new_key)
        positions = random.uniform(
            subkey,
            shape=(self.nums_per_image, 2),
            minval=0,
            maxval=jnp.array((self.lims[0], self.lims[1])),
        )
        new_state = indexes, positions, velocs

        return new_state, new_key

    # @partial(jit, static_argnums=(0,))
    def reset(self, key):
        env_state = self._reset(key)
        initial_state = env_state[0]
        return env_state, self._get_obsv(initial_state)


with jax.checking_leaks():
    seed = 42
    key = random.key(seed)
    test_images = mnist.test_images()
    env = JaxMNISTLoader(test_images)
    env_state, inital_obsv = env.reset(key)
    action = 1
    for i in range(100):
        env_state, obsv, reward, done, info = env.step(env_state, 1)
        fig = np.array(obsv, dtype="int")
        # imageio.imwrite(os.path.join("./figs", f"mmnist_{i}.png"), fig)
        matplotlib.image.imsave(
            os.path.join("./temp", f"mmnist_{i}.png"), fig, cmap="grey"
        )
    print(obsv)
