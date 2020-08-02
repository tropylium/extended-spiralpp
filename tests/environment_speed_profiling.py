# Copyright urw7rs.
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
import traceback

import argparse
import multiprocessing as mp

import nest
import torch
import libtorchbeast

import numpy as np

from gym import spaces

sys.path.append("..")
from torchbeast import polybeast_learner
from torchbeast import polybeast_env
from torchbeast import utils

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

# batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def run_env(flags, actor_id):
    np.random.seed()  # Get new random seed in forked process.
    polybeast_env.main(flags)


def main():
    filename = "environment_speed_test.json"
    with torch.autograd.profiler.profile() as prof:
        run()
        logging.info("Collecting trace and writing to '%s.gz'", filename)
    prof.export_chrome_trace(filename)
    os.system("gzip %s" % filename)


def run():
    flags = argparse.Namespace()
    flags, argv = polybeast_learner.parser.parse_known_args(namespace=flags)
    flags, argv = polybeast_env.parser.parse_known_args(args=argv, namespace=flags)
    if argv:
        # Produce an error message.
        polybeast_env.parser.print_usage()
        print("Unkown args:", " ".join(argv))
        return -1

    env_processes = []
    for actor_id in range(1):
        p = mp.Process(target=run_env, args=(flags, actor_id))
        p.start()
        env_processes.append(p)

    if flags.max_learner_queue_size is None:
        flags.max_learner_queue_size = flags.batch_size

    learner_queue = libtorchbeast.BatchingQueue(
        batch_dim=1,
        minimum_batch_size=flags.batch_size,
        maximum_batch_size=flags.batch_size,
        check_inputs=True,
        maximum_queue_size=flags.max_learner_queue_size,
    )
    replay_queue = libtorchbeast.BatchingQueue(
        batch_dim=1,
        minimum_batch_size=1,
        maximum_batch_size=flags.num_actors,
        timeout_ms=100,
        check_inputs=True,
        maximum_queue_size=flags.num_actors,
    )
    inference_batcher = libtorchbeast.DynamicBatcher(
        batch_dim=1,
        minimum_batch_size=1,
        maximum_batch_size=512,
        timeout_ms=100,
        check_outputs=True,
    )

    addresses = []
    connections_per_server = 1
    pipe_id = 0
    while len(addresses) < flags.num_actors:
        for _ in range(connections_per_server):
            addresses.append(f"{flags.pipes_basename}.{pipe_id}")
            if len(addresses) == flags.num_actors:
                break
        pipe_id += 1

    actors = libtorchbeast.ActorPool(
        unroll_length=flags.unroll_length,
        learner_queue=learner_queue,
        replay_queue=replay_queue,
        inference_batcher=inference_batcher,
        env_server_addresses=addresses,
        initial_agent_state=(),
    )

    def run():
        try:
            actors.run()
            print("actors are running")
        except Exception as e:
            logging.error("Exception in actorpool thread!")
            traceback.print_exc()
            print()
            raise e

    actorpool_thread = threading.Thread(target=run, name="actorpool-thread")

    def dequeue(queue):
        for tensor in queue:
            del tensor

    dequeue_threads = [
        threading.Thread(target=dequeue, name="dequeue-thread-%i" % i, args=(queue,))
        for i, queue in enumerate([learner_queue, replay_queue])
    ]

    # create an environment to sample random actions
    dataset_uses_color = flags.dataset not in ["mnist", "omniglot"]
    grayscale = dataset_uses_color and not flags.use_color

    is_color = flags.use_color or flags.env_type == "fluid"
    if is_color is False:
        grayscale = True
    else:
        grayscale = is_color and not dataset_uses_color

    env_name, config = utils.parse_flags(flags)
    env = utils.create_env(env_name, config, grayscale, dataset=None)

    if flags.condition:
        new_space = env.observation_space.spaces
        c, h, w = new_space["canvas"].shape
        new_space["canvas"] = spaces.Box(
            low=0, high=255, shape=(c * 2, h, w), dtype=np.uint8
        )
        env.observation_space = spaces.Dict(new_space)

    action_space = env.action_space
    env.close()

    def inference(inference_batcher, lock=threading.Lock()):
        nonlocal step

        for batch in inference_batcher:
            batched_env_outputs, agent_state = batch.get_inputs()

            obs, _, done, *_ = batched_env_outputs
            B = done.shape[1]

            with lock:
                step += B

            actions = nest.map(lambda i: action_space.sample(), [i for i in range(B)])
            action = torch.from_numpy(np.concatenate(actions)).view(1, B, -1)

            outputs = ((action,), ())
            outputs = nest.map(lambda t: t, outputs)
            batch.set_outputs(outputs)

    lock = threading.Lock()
    inference_threads = [
        threading.Thread(
            target=inference,
            name="inference-thread-%i" % i,
            args=(inference_batcher, lock),
        )
        for i in range(flags.num_inference_threads)
    ]

    actorpool_thread.start()

    threads = dequeue_threads + inference_threads

    for t in threads:
        t.start()

    step = 0

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

    inference_batcher.close()
    learner_queue.close()

    replay_queue.close()

    actorpool_thread.join()

    for t in threads:
        t.join()

    for p in env_processes:
        p.terminate()


if __name__ == "__main__":
    main()
