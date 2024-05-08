from datasets import JaxMMNIST


def load_dataset(cfg, **kwargs):
    if cfg.dataset == "jax_mmnist":

        if cfg.device == "cuda:0":
            device = 0
        elif cfg.device == "cuda:1":
            device = 1
        else:
            print("Not supported device for jax")
            exit(1)

        train_data = JaxMMNIST(
            train=True,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
            num_source_mnist_images=cfg.num_mnist_patch,
            device=device,
        )
        test_data = JaxMMNIST(
            train=False,
            seq_len=cfg.eval_seq_len,
            batch_size=cfg.eval_batch_size,
            num_source_mnist_images=cfg.num_mnist_patch,
            device=device,
        )
    else:
        raise ValueError("Dataset {} not supported.".format(cfg.dataset))

    return train_data, test_data
