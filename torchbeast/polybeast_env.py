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

from torch.utils.data import Subset
import libtorchbeast

from torchbeast import utils

# yapf: disable
parser = argparse.ArgumentParser(description='Remote Environment Server')

parser.add_argument("--pipes_basename", default="unix:/tmp/polybeast",
                    help="Basename for the pipes for inter-process communication. "
                    "Has to be of the type unix:/some/path.")
parser.add_argument("--num_actors", default=4, type=int, metavar='N',
                    help='Number of environment actors(servers).')

BRUSHES_BASEDIR = os.path.join(os.getcwd(), "third_party/mypaint-brushes-1.3.0")
BRUSHES_BASEDIR = os.path.abspath(BRUSHES_BASEDIR)

SHADERS_BASEDIR = os.path.join(os.getcwd(), "third_party/paint/shaders")
SHADERS_BASEDIR = os.path.abspath(SHADERS_BASEDIR)

parser.add_argument("--env_type", type=str, default="libmypaint",
                    help="Environment. Ignored if --no_start_servers is passed.")
parser.add_argument("--episode_length", type=int, default=20,
                    help="Set epiosde length")
parser.add_argument("--canvas_width", type=int, default=256, metavar="W",
                    help="Set canvas render width")
parser.add_argument("--brush_type", type=str, default="classic/dry_brush",
                    help="Set brush type from brush dir")
parser.add_argument("--brush_sizes", nargs='+', type=int,
                    default=[1, 2, 4, 8, 12, 24],
                    help="Set brush_sizes float is allowed")
parser.add_argument("--use_color", action="store_true",
                    help="use_color flag")
parser.add_argument("--use_pressure", action="store_true",
                    help="use_pressure flag")
parser.add_argument("--use_compound", action="store_true",
                    help="use compound action space")
parser.add_argument("--new_stroke_penalty", type=float, default=0.0,
                    help="penalty for new stroke")
parser.add_argument("--stroke_length_penalty", type=float, default=0.0,
                    help="penalty for stroke length")
parser.add_argument("--condition", action="store_true",
                    help='condition flag')
parser.add_argument("--dataset",
                    help="Dataset name. MNIST, Omniglot, CelebA, CelebA-HQ is supported")

# yapf: enable


def serve(env_name, config, grayscale, dataset, server_address):
    init = lambda: utils.create_env(env_name, config, grayscale, dataset)
    server = libtorchbeast.Server(init, server_address=server_address)
    server.run()


def main(flags):
    env_name, config = utils.parse_flags(flags)

    if not flags.pipes_basename.startswith("unix:"):
        raise Exception("--pipes_basename has to be of the form unix:/some/path.")

    dataset_uses_color = flags.dataset not in ["mnist", "omniglot"]
    grayscale = dataset_uses_color and not flags.use_color

    if flags.condition:
        dataset = utils.create_dataset(flags.dataset, grayscale)
        per_actor = len(dataset) // flags.actors

    is_color = flags.use_color or flags.env_type == "fluid"
    if is_color is False:
        grayscale = True
    else:
        grayscale = is_color and not dataset_uses_color

    processes = []
    for i in range(flags.num_actors):
        if flags.condition:
            start = per_actor * i
            end = min(start + per_actor, len(dataset))

            dataset = Subset(dataset, range(start, end + 1))
        else:
            dataset = None

        p = mp.Process(
            target=serve,
            args=(env_name, config, grayscale, dataset, f"{flags.pipes_basename}.{i}"),
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


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
