import einops
import mnist
from jax_loader import JaxMNISTLoader
import jax


def JaxMMNIST(
    train=True,
    batch_size=256,
    num_source_mnist_images=1000,
    num_mnist_per_image=2,
    seq_len=100,
    device=0,
):

    if train:
        mnist_images = mnist.train_images()
    else:
        mnist_images = mnist.test_images()

    np_images = mnist_images[:num_source_mnist_images]
    jax_images = jax.device_put(np_images, jax.devices()[device])

    jaxLoader = JaxMNISTLoader(
        images=jax_images, seq_len=seq_len, num_mnist_per_mmnist=num_mnist_per_image
    )

    seed = 38
    next_key = jax.random.key(seed)

    batch_build_seq = jax.jit(
        jax.vmap(jaxLoader.build_seq),
    )

    while True:
        next_key, current_key = jax.random.split(next_key)
        batch_key = jax.random.split(current_key, num=batch_size)
        batch_ys = batch_build_seq(batch_key)
        reshaped_ys = einops.rearrange(batch_ys, "b t w h -> b t w h 1")
        yield reshaped_ys
