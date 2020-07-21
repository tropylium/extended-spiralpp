import os
from filelock import FileLock

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import CelebA, Omniglot, MNIST

from torchbeast import env_wrapper
from torchbeast.core.datasets import CelebAHQ

frame_width = 64
grid_width = 32


BRUSHES_BASEDIR = os.path.join(os.getcwd(), "third_party/mypaint-brushes-1.3.0")
BRUSHES_BASEDIR = os.path.abspath(BRUSHES_BASEDIR)

SHADERS_BASEDIR = os.path.join(os.getcwd(), "third_party/paint/shaders")
SHADERS_BASEDIR = os.path.abspath(SHADERS_BASEDIR)


def parse_flags(flags):
    config = dict(
        episode_length=flags.episode_length,
        canvas_width=flags.canvas_width,
        grid_width=grid_width,
        brush_sizes=flags.brush_sizes,
    )

    if flags.env_type == "libmypaint":
        config.update(
            dict(
                brush_type=flags.brush_type,
                use_pressure=flags.use_pressure,
                use_color=flags.use_color,
                use_alpha=False,
                background="white",
                brushes_basedir=BRUSHES_BASEDIR,
            )
        )
        env_name = "Libmypaint"

    elif flags.env_type == "fluid":
        config["shaders_basedir"] = SHADERS_BASEDIR
        env_name = "Fluid"

    if flags.use_compound:
        config.update(
            dict(
                new_stroke_penalty=flags.new_stroke_penalty,
                stroke_length_penalty=flags.stroke_length_penalty,
            )
        )
        env_name += "-v1"

    else:
        env_name += "-v0"

    return env_name, config


def create_dataset(name, grayscale, normalize=True):
    tsfm = []
    if grayscale:
        tsfm.append(transforms.Grayscale())
    tsfm.extend([transforms.Resize((frame_width, frame_width)), transforms.ToTensor()])

    with FileLock("./dataset.lock"):
        if name == "mnist":
            if normalize:
                tsfm.append(transforms.Normalize((0.5,), (0.5,)))
            tsfm = transforms.Compose(tsfm)
            dataset = MNIST(root="./", train=True, transform=tsfm, download=True)

        elif name == "omniglot":
            if normalize:
                tsfm.append(transforms.Normalize((0.5,), (0.5,)))
            tsfm = transforms.Compose(tsfm)
            dataset = Omniglot(
                root="./", background=True, transform=tsfm, download=True
            )

        elif name == "celeba":
            if normalize:
                tsfm.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            tsfm = transforms.Compose(tsfm)
            dataset = CelebA(
                root="./",
                split="train",
                target_type=None,
                transform=tsfm,
                download=True,
            )

        elif name == "celeba-hq":
            if normalize:
                tsfm.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            tsfm = transforms.Compose(tsfm)
            dataset = CelebAHQ(root="./", split="train", transform=tsfm, download=True)

        else:
            raise NotImplementedError

    return dataset


def create_env(env_name, config, grayscale, dataset):
    env = env_wrapper.make_raw(env_name, config)

    if frame_width != config["canvas_width"]:
        env = env_wrapper.WarpFrame(
            env,
            width=frame_width,
            height=frame_width,
            grayscale=grayscale,
            dict_space_key="canvas",
        )

    env = env_wrapper.Base(env)

    if isinstance(dataset, Dataset):
        env = env_wrapper.ConcatTarget(env, dataset)

    return env
