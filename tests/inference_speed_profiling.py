# Copyright (c) Facebook, Inc. and its affiliates.
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

import logging
import os
import sys
import threading
import time
import timeit

import torch

import nest

sys.path.append("..")
from torchbeast.core import models as experiment

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4
num_inference_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def main():
    filename = "inference_speed_test.json"
    with torch.autograd.profiler.profile() as prof:
        run()
        logging.info("Collecting trace and writing to '%s.gz'", filename)
    prof.export_chrome_trace(filename)
    os.system("gzip %s" % filename)


def run():
    size = (3, 64, 64)
    action_shape = [1024, 1024, 2, 6, 20, 20, 20]
    grid_shape = [32, 32]

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = experiment.Net(
        obs_shape=size, action_shape=action_shape, grid_shape=grid_shape,
    )
    model = model.to(device=device)

    should_stop = threading.Event()

    step = 0

    def stream_inference(input):
        nonlocal step

        obs, *_ = input
        T, B, *_ = obs["canvas"].shape
        stream = torch.cuda.Stream()

        with torch.no_grad():
            with torch.cuda.stream(stream):
                while not should_stop.is_set():
                    input = nest.map(lambda t: t.pin_memory(), input)
                    input = nest.map(lambda t: t.to(device), input)
                    outputs = model(*input)
                    outputs = [t.cpu() for t in outputs]
                    stream.synchronize()
                    step += B

    def inference(input, lock=threading.Lock()):  # noqa: B008
        nonlocal step

        obs, *_ = input
        T, B, *_ = obs["canvas"].shape
        with torch.no_grad():
            while not should_stop.is_set():
                input = nest.map(lambda t: t.to(device), input)
                with lock:
                    outputs = model(*input)
                    step += B
                outputs = nest.map(lambda t: t.cpu(), outputs)

    def direct_inference(input):
        nonlocal step
        input = nest.map(lambda t: t.to(device), input)

        obs, *_ = input
        T, B, *_ = obs["canvas"].shape
        with torch.no_grad():
            while not should_stop.is_set():
                model(*input)
                step += B

    obs = dict(
        canvas=torch.randn(1, batch_size, *size),
        prev_action=torch.ones(1, batch_size, len(action_shape)),
        action_mask=torch.ones(1, batch_size, len(action_shape)),
        noise_sample=torch.randn(1, batch_size, 10),
    )
    done = torch.zeros(1, batch_size, dtype=torch.bool)
    core_state = model.initial_state(batch_size)

    input = (obs, done, core_state)
    work_threads = [
        threading.Thread(target=stream_inference, args=(input,))
        for _ in range(num_inference_threads)
    ]
    for thread in work_threads:
        thread.start()

    try:
        while step < 10000:
            start_time = timeit.default_timer()
            start_step = step
            time.sleep(3)
            end_step = step

            logging.info(
                "Step %i @ %.1f SPS.",
                end_step,
                (end_step - start_step) / (timeit.default_timer() - start_time),
            )
    except KeyboardInterrupt:
        pass

    should_stop.set()
    for thread in work_threads:
        thread.join()


if __name__ == "__main__":
    main()
