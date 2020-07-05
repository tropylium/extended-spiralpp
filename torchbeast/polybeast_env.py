# Copyright (c) Facebook, Inc. and its affiliates.
# 2 May 2020 - Modified by urw7rs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import multiprocessing as mp
import time
import ast

import torchvision.transforms as transforms
from torchvision.datasets import CelebA, Omniglot, MNIST
from torch.utils.data import Subset

from torchbeast import env_wrapper
from torchbeast.core import datasets
from libtorchbeast import rpcenv


# yapf: disable
parser = argparse.ArgumentParser(description='Remote Environment Server')

parser.add_argument("--pipes_basename", default="unix:/tmp/polybeast",
                    help="Basename for the pipes for inter-process communication. "
                    "Has to be of the type unix:/some/path.")
parser.add_argument("--num_servers", default=4, type=int, metavar='N',
                    help='Number of environment servers.')

BRUSHES_BASEDIR = os.path.join(os.getcwd(), "third_party/mypaint-brushes-1.3.0")
BRUSHES_BASEDIR = os.path.abspath(BRUSHES_BASEDIR)

SHADERS_BASEDIR = os.path.join(os.getcwd(), "third_party/paint/shaders")
SHADERS_BASEDIR = os.path.abspath(SHADERS_BASEDIR)

parser.add_argument("--env", type=str, default="Libmypaint-v0",
                    help="Environment. Ignored if --no_start_servers is passed.")
parser.add_argument("--env_type", type=str, default="libmypaint",
                    help="Environment. Ignored if --no_start_servers is passed.")
parser.add_argument("--episode_length", type=int, default=20,
                    help="Set epiosde length")
parser.add_argument("--canvas_width", type=int, default=256,
                    help="Set canvas render width")
parser.add_argument("--brush_type", type=str, default="classic/dry_brush",
                    help="Set brush type from brush dir")
parser.add_argument("--brush_sizes",
                    default=[1, 2, 4, 6, 12, 24],
                    help="Set brush_sizes float is allowed")
parser.add_argument("--use_color", action="store_true",
                    help="use_color flag")
parser.add_argument("--use_pressure", action="store_true",
                    help="use_pressure flag")
parser.add_argument("--use_alpha", action="store_true",
                    help="use_alpha flag")
parser.add_argument("--background", type=str, default="white",
                    help="Set background color [white, transparent]")
parser.add_argument("--brushes_basedir", type=str, default=BRUSHES_BASEDIR,
                    help="Set brush base path")
parser.add_argument("--shaders_basedir", type=str, default=SHADERS_BASEDIR,
                    help="Set shader base path")
parser.add_argument("--new_stroke_penalty", type=float, default=0.0,
                    help="penalty for new stroke")
parser.add_argument("--stroke_length_penalty", type=float, default=0.0,
                    help="penalty for stroke length")
parser.add_argument("--condition", action="store_true",
                    help='condition flag')
parser.add_argument("--dataset",
                    help="Dataset name. MNIST, Omniglot, CelebA, CelebA-HQ is supported")
parser.add_argument("--num_actors", type=int, metavar="N",
                    help="Number of actors.")

# yapf: enable

frame_width = 64
grid_width = 32


def create_env(env_name, config):
    keys = [
        "new_stroke_penalty",
        "stroke_length_penalty",
        "dataset",
        "num_actors",
        "actor_id",
    ]
    new_stroke, stroke_length, dataset, num_actors, actor_id = (
        config.pop(k, None) for k in keys
    )
    condition = flags.condition

    env = env_wrapper.make_raw(env_name, config)

    if new_stroke is not None:
        env.set_new_stroke_penalty(new_stroke)
    if stroke_length is not None:
        env.set_stroke_length_penalty(stroke_length)

    if frame_width != flags.canvas_width:
        env = env_wrapper.WarpFrame(env, width=frame_width, height=frame_width)

    env = env_wrapper.Base(env)

    if condition:
        per_actor = len(dataset) // num_actors
        start = per_actor * actor_id
        end = min(start + per_actor, len(dataset))

        dataset = Subset(dataset, range(start, end + 1))

        env = env_wrapper.ConcatTarget(env, dataset)

    return env


def serve(env_name, config, server_address):
    init = lambda: create_env(env_name, config)
    server = rpcenv.Server(init, server_address=server_address)
    server.run()


if __name__ == "__main__":
    flags = parser.parse_args()

    if isinstance(flags.brush_sizes, str):
        flags.brush_sizes = ast.literal_eval(flags.brush_sizes)

    config = dict(
        episode_length=flags.episode_length,
        canvas_width=flags.canvas_width,
        grid_width=grid_width,
        brush_sizes=flags.brush_sizes,
    )

    if flags.env_type == "fluid":
        config["shaders_basedir"] = flags.shaders_basedir
    elif flags.env_type == "libmypaint":
        config.update(
            dict(
                brush_type=flags.brush_type,
                use_color=flags.use_color,
                use_pressure=flags.use_pressure,
                use_alpha=flags.use_alpha,
                background=flags.background,
                brushes_basedir=flags.brushes_basedir,
            )
        )

    if flags.env.split("-")[1] == "v1":
        config.update(
            dict(
                new_stroke_penalty=flags.new_stroke_penalty,
                stroke_length_penalty=flags.stroke_length_penalty,
            )
        )

    if flags.condition:
        tsfm = transforms.Compose(
            [transforms.Resize((frame_width, frame_width)), transforms.ToTensor()]
        )

        dataset = flags.dataset

        if dataset == "mnist":
            dataset = MNIST(root="./", train=True, transform=tsfm, download=True)
        elif dataset == "omniglot":
            dataset = Omniglot(
                root="./", background=True, transform=tsfm, download=True
            )
        elif dataset == "celeba":
            dataset = CelebA(
                root="./",
                split="train",
                target_type=None,
                transform=tsfm,
                download=True,
            )
        elif dataset == "celeba-hq":
            dataset = datasets.CelebAHQ(
                root="./", split="train", transform=tsfm, download=True
            )
        else:
            raise NotImplementedError

        config["dataset"] = dataset
        config["num_actors"] = flags.num_actors

    if not flags.pipes_basename.startswith("unix:"):
        raise Exception("--pipes_basename has to be of the form unix:/some/path.")

    processes = []
    for i in range(flags.num_servers):
        config["actor_id"] = i

        p = mp.Process(
            target=serve,
            args=(flags.env, config, f"{flags.pipes_basename}.{i}",),
            daemon=True,
        )
        p.start()
        processes.append(p)

    try:
        # We are only here to listen to the interrupt.
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        pass
