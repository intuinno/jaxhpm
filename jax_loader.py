import jax.numpy as jnp
import jax
import numpy as np
from jax import random
import mnist
import os
import matplotlib.image
import jax.profiler


class JaxMNISTLoader:

    def __init__(self, images=None, seq_len=100, num_mnist_per_mmnist=2):
        self.width, self.height = 64, 64
        self.mnist_width, self.mnist_height = 28, 28
        self.lims = (x_lim, y_lim) = (
            self.width - self.mnist_width,
            self.height - self.mnist_height,
        )
        self.images = images
        self.num_images = images.shape[0]
        self.seq_len = seq_len
        self.nums_per_image = num_mnist_per_mmnist

    # @partial(jit, static_argnums=(0,))
    def step(self, env_state, action):
        state, key = env_state
        indexes, positions, velocs = state
        next_pos = positions + velocs

        velocs = jax.lax.select(
            ((next_pos < -2) | (next_pos > self.lims[0] + 2)), -1.0 * velocs, velocs
        )

        positions = positions + velocs
        next_state = (indexes, positions, velocs)
        next_env_state = next_state, key
        reward = 1.0
        done = False
        return next_env_state, self._get_obsv(next_state), reward, done, {}

    # @partial(jit, static_argnums=(0,))
    def _build_canvas(self, index, position):
        x, y = position.astype(int)
        x = jnp.where(x < 0, 0, x)
        x = jnp.where(x > self.lims[0], self.lims[0], x)
        y = jnp.where(y < 0, 0, y)
        y = jnp.where(y > self.lims[1], self.lims[1], y)
        image = self.images[index]

        canvas = jnp.zeros((self.width, self.height), dtype="uint8")
        fig = jax.lax.dynamic_update_slice(canvas, image, (x, y))

        return fig

    # @partial(jit, static_argnums=(0,))
    def _get_obsv(self, state):
        indexes, positions, _ = state
        canvas = jnp.zeros((self.width, self.height))
        for i, p in zip(indexes, positions):
            canvas += self._build_canvas(i, p)
        canvas = jnp.where(canvas > 255.0, 255.0, canvas)
        canvas /= 255.0
        return canvas

    def _maybe_reset(self, env_state, done):
        key = env_state[1]
        return jax.lax.cond(done, self._reset, lambda key: env_state, key)

    def _reset(self, key):
        new_key, subkey = random.split(key)
        direcs = jnp.pi * (random.uniform(subkey, shape=(self.nums_per_image,)) * 2 - 1)

        new_key, subkey = random.split(new_key)
        indexes = random.randint(subkey, (self.nums_per_image,), 0, self.num_images)

        new_key, subkey = random.split(new_key)
        speeds = random.randint(subkey, (self.nums_per_image,), 0, 5) + 2
        velocs = jnp.array(
            [
                (speed * jnp.cos(direc), speed * jnp.sin(direc))
                for direc, speed in zip(direcs, speeds)
            ]
        )

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

    # @partial(jit, static_argnums=(0,))
    def scan_func(self, carry, x):
        new_carry, y, _, _, _ = self.step(carry, x)
        return new_carry, y

    # @partial(jit, static_argnums=(0,))
    def build_seq(self, key):
        init, _ = self.reset(key)
        _, ys = jax.lax.scan(self.scan_func, init, None, length=self.seq_len)
        return ys


if __name__ == "__main__":

    # with jax.profiler.trace("./logs"):

    seq_len = 256
    seed = 42
    num_mnist_images = 1000
    batch_size = 128

    key = random.key(seed)
    all_images = mnist.test_images()
    np_images = all_images[:num_mnist_images]
    jax_images = jnp.array(np_images)

    env = JaxMNISTLoader(seq_len=seq_len, images=jax_images)

    env_state, _ = env.reset(key)

    action = 1
    for i in range(seq_len):
        env_state, obsv, reward, done, info = env.step(env_state, 1)
        fig = np.array(obsv, dtype="int")
        matplotlib.image.imsave(
            os.path.join("./temp", f"mmnist_{i}.png"), fig, cmap="grey"
        )
    print(obsv)

    batch_key = random.split(key, num=batch_size)

    batch_build_seq = jax.jit(jax.vmap(env.build_seq))
    batch_ys = batch_build_seq(batch_key)
    print(batch_ys.shape)
    batch_ys.block_until_ready()
    jax.profiler.save_device_memory_profile("memory.prof")
