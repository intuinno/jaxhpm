import numpy as np
import argparse
from ruamel.yaml import YAML
import pathlib
import sys
from tqdm import tqdm

# from cwvae import build_model
# from loggers.summary import Summary
# from loggers.checkpoint import Checkpoint
from data_loader import *
import tools

# from l2hwm import L2HWM
from jaxhpm import JaxHPM
from datetime import datetime
import pytz
import jax
import ninjax as nj
import embodied


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--load_model", type=str)
    args, remaining = parser.parse_known_args()
    rootdir = pathlib.Path(sys.argv[0]).parent
    yaml = YAML(typ="safe", pure=True)
    configs = yaml.load((rootdir / "configs.yml").read_text())

    defaults = {}
    for name in args.configs:
        defaults.update(configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    configs = parser.parse_args(remaining)

    exp_name = configs.exp_name + "_"
    tools.set_seed_everywhere(configs.seed)

    tz = pytz.timezone("US/Central")
    now = datetime.now(tz)
    date_time = now.strftime("%Y%m%d_%H%M%S")
    exp_name += date_time

    # Creating model dir with experiment name.
    exp_logdir = rootdir / configs.logdir / configs.dataset / exp_name
    print("Logdir", exp_logdir)
    exp_logdir.mkdir(parents=True, exist_ok=True)

    # Dumping config.
    yaml.default_flow_style = False
    with open(exp_logdir / "config.yml", "w") as f:
        yaml.dump(vars(configs), f)

    # Change to Config
    configs = embodied.core.Config(vars(configs))

    # Load dataset.
    train_dataloader, val_dataloader = load_dataset(configs)

    # Build model
    model = JaxHPM(configs, name="jaxhpm")

    pure_init = nj.pure(lambda x: model.initial(len(x)))
    pure_train = nj.pure(model.train)
    jit_init = nj.jit(pure_init)
    jit_train = nj.jit(pure_train)

    print(f"========== Using {configs.device} device ===================")

    # Load model if args.load_model is not none
    if args.load_model is not None:
        model_path = pathlib.Path(args.load_model).expanduser()
        print(f"========== Loading saved model from {model_path} ===========")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Build logger
    logger = tools.Logger(exp_logdir, 0)
    metrics = {}

    num_val_enumerate = configs.val_num_mmnist_seq // configs.eval_batch_size
    num_train_enumerate = configs.train_num_mmnist_seq // configs.batch_size

    x = next(train_dataloader)
    state = {}
    state = nj.init(model.local_train)(state, x, seed=42)
    local_train = nj.pure(model.local_train)
    local_train = jax.jit(local_train)

    for epoch in range(configs.num_epochs):
        # Write evaluation summary
        # print(f"======== Epoch {epoch} / {configs.num_epochs} ==========")
        # now = datetime.now(tz)
        # current_time = now.strftime("%H:%M:%S")
        # print("Current Time =", current_time)

        # logger.step = epoch

        # if epoch % configs.eval_every == 0:
        #     print(f"Evaluating ... ")
        #     recon_loss_list = []
        #     for i, _ in enumerate(tqdm(range(num_val_enumerate))):
        #         x = next(val_dataloader)
        #         openl, recon_loss = model.video_pred(x.to(configs.device))
        #         if i == 0:
        #             logger.video("eval_openl", openl)
        #         recon_loss_list.append(recon_loss)
        #     recon_loss_mean = np.mean(recon_loss_list)
        #     logger.scalar("eval_video_nll", recon_loss_mean)

        # with jax.profiler.trace("./logs"):
        print(f"Training ...")
        for i, _ in enumerate(tqdm(range(num_train_enumerate))):
            x = next(train_dataloader)
            state, met = local_train(state, x)
            for name, values in met.items():
                if not name in metrics.keys():
                    metrics[name] = [values]
                else:
                    metrics[name].append(values)

        # Write training summary
        for name, values in metrics.items():
            logger.scalar(name, float(np.mean(values)))
            metrics[name] = []
        if epoch % configs.train_gif_every == 0:
            openl, recon_loss = model.video_pred(x)
            logger.video("train_openl", openl)
            logger.write(fps=True)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            # 'logger': logger,
        }
        # Save Check point
        if epoch % configs.save_model_every == 0:
            torch.save(checkpoint, exp_logdir / "latest_checkpoint.pt")

        if epoch % configs.backup_model_every == 0:
            torch.save(checkpoint, exp_logdir / f"state_{epoch}.pt")

    print("Training complete.")
