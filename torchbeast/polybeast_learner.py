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

import argparse
import collections
import logging
import os
import threading
import time
import timeit
import traceback
import random

os.environ["OMP_NUM_THREADS"] = "4"  # Necessary for multithreading.

import nest
import torch
import torch.optim as optim
import libtorchbeast
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import numpy as np

from gym import spaces

from torchbeast import utils
from torchbeast.core import file_writer
from torchbeast.core import vtrace
from torchbeast.core import models

# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--pipes_basename", default="unix:/tmp/polybeast",
                    help="Basename for the pipes for inter-process communication. "
                    "Has to be of the type unix:/some/path.")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")
parser.add_argument("--start_servers", dest="start_servers", action="store_true",
                    help="Spawn polybeast_env servers automatically.")
parser.add_argument("--no_start_servers", dest="start_servers", action="store_false",
                    help="Don't spawn polybeast_env servers automatically.")
parser.set_defaults(start_servers=True)

# Environment settings
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

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="~/logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=4, type=int, metavar="N",
                    help="Number of actors.")
parser.add_argument("--total_steps", default=100000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=64, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--num_learner_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--num_inference_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--replay_buffer_size", default=None, type=int, metavar="N",
                    help="Replay buffer size. Defaults to batch_size * 20.")
parser.add_argument("--max_learner_queue_size", default=None, type=int, metavar="N",
                    help="Optional maximum learner queue size. Defaults to batch_size.")
parser.add_argument("--unroll_length", default=20, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--condition", action="store_true",
                    help='condition flag')
parser.add_argument("--use_tca", action="store_true",
                    help="temporal credit assignment flag")
parser.add_argument("--dataset", default="celeba-hq",
                    help="Dataset name. MNIST, Omniglot, CelebA, CelebA-HQ is supported")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.01, type=float,
                    help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5, type=float,
                    help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99, type=float,
                    help="Discounting factor.")
parser.add_argument("--gp_weight", default=10.0, type=float,
                    help="Gradient penalty weight.")

# Optimizer settings.
parser.add_argument("--policy_learning_rate", default=0.0003, type=float,
                    metavar="LRP", help="Policy learning rate.")
parser.add_argument("--discriminator_learning_rate", default=0.0001, type=float,
                    metavar="LRD", help="Discriminator learning rate.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")

# Misc settings.
parser.add_argument("--write_profiler_trace", action="store_true",
                    help="Collect and write a profiler trace "
                    "for chrome://tracing/.")

# yapf: enable

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

frame_width = 64
grid_width = 32


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    entropy = 0
    for logit in logits:
        policy = F.softmax(logit, dim=-1)
        log_policy = F.log_softmax(logit, dim=-1)
        entropy += torch.sum(policy * log_policy)
    return entropy


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = 0.0
    for i, logit in enumerate(logits):
        cross_entropy = cross_entropy + F.nll_loss(
            F.log_softmax(torch.flatten(logit, 0, 1), dim=-1),
            target=torch.flatten(actions[..., i].long(), 0, 1),
            reduction="none",
        )

    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []

        self.capacity = capacity
        self.position = 0

    def push(self, frame):
        frames = frame.split(1)
        request = len(frames)

        free = self.capacity - self.position
        available = len(self.buffer) - self.position
        if available < free:
            if request > self.capacity:
                size = free
            else:
                size = request
            self.buffer.extend([None for _ in range(size)])

        if request > free:
            self.buffer[self.position :] = frames[:free]

            frames = frames[free:]

            self.position = 0

            request -= free

        self.buffer[self.position : self.position + request] = frames
        self.position = (self.position + request) % self.capacity

    def sample(self, batch_size):
        frames = random.sample(self.buffer, batch_size)

        return torch.cat(frames)

    def __len__(self):
        return len(self.buffer)

    def checkpoint(self):
        return dict(
            buffer=torch.cat(self.buffer),
            position=self.position,
            capacity=self.capacity,
        )

    def load_checkpoint(self, checkpoint):
        self.buffer = list(checkpoint["buffer"].split(1))
        self.position = checkpoint["position"]
        self.capacity = checkpoint["capacity"]


def inference(flags, inference_batcher, model, lock=threading.Lock()):
    with torch.no_grad():
        for batch in inference_batcher:
            batched_env_outputs, agent_state = batch.get_inputs()

            obs, _, done, *_ = batched_env_outputs

            obs, done, agent_state = nest.map(
                lambda t: t.to(flags.actor_device, non_blocking=True),
                [obs, done, agent_state],
            )

            with lock:
                outputs = model(obs, done, agent_state)

            outputs = nest.map(lambda t: t.cpu(), outputs)
            batch.set_outputs(outputs)


EnvOutput = collections.namedtuple(
    "EnvOutput", "obs, reward, done, episode_step episode_return"
)
AgentOutput = collections.namedtuple("AgentOutput", "action policy_logits baseline")
Batch = collections.namedtuple("Batch", "env agent")


def tca_reward_function(flags, obs, new_frame, D):
    frame = obs["canvas"][:-1]
    frame, new_frame = nest.map(lambda t: torch.flatten(t, 0, 1), (frame, new_frame))

    with torch.no_grad():
        reward = torch.zeros(
            flags.unroll_length + 1, flags.batch_size, device=flags.learner_device
        )

        reward[1:] += (D(new_frame) - D(frame)).view(-1, flags.batch_size)

    return reward


def reward_function(flags, done, new_frame, D):
    index = done[1:].nonzero(as_tuple=False)
    new_frame = new_frame[index[:, 0], index[:, 1]]

    with torch.no_grad():
        reward = torch.zeros(
            flags.unroll_length + 1, flags.batch_size, device=flags.learner_device
        )

        reward[index[:, 0], index[:, 1]] += D(new_frame)

    return reward


def edit_tuple(old_tuple, index, data):
    list_tuple = list(old_tuple)
    list_tuple[index] = data
    new_tuple = tuple(list_tuple)
    return new_tuple


def learn(
    flags,
    learner_queue,
    model,
    actor_model,
    D,
    optimizer,
    scheduler,
    stats,
    plogger,
    lock=threading.Lock(),
):
    for tensors in learner_queue:
        new_obs = tensors[1]
        tensors = edit_tuple(tensors, 1, new_obs["canvas"])

        tensors = nest.map(
            lambda t: t.to(flags.learner_device, non_blocking=True), tensors
        )

        batch, new_frame, initial_agent_state = tensors

        env_outputs, actor_outputs = batch
        obs, reward, done, step, _ = env_outputs

        lock.acquire()  # Only one thread learning at a time.

        if flags.use_tca:
            discriminator_reward = tca_reward_function(flags, obs, new_frame, D)

            reward = env_outputs[1]
            env_outputs = edit_tuple(env_outputs, 1, reward + discriminator_reward)
            batch = edit_tuple(batch, 0, env_outputs)
        else:
            if done.any().item():
                discriminator_reward = reward_function(flags, done, new_frame, D)

                reward = env_outputs[1]
                env_outputs = edit_tuple(env_outputs, 1, reward + discriminator_reward)
                batch = edit_tuple(batch, 0, env_outputs)

        optimizer.zero_grad()

        actor_outputs = AgentOutput._make(actor_outputs)

        learner_outputs, agent_state = model(obs, done, initial_agent_state)

        # Take final value function slice for bootstrapping.
        learner_outputs = AgentOutput._make(learner_outputs)
        bootstrap_value = learner_outputs.baseline[-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = nest.map(lambda t: t[1:], batch)
        learner_outputs = nest.map(lambda t: t[:-1], learner_outputs)

        # Turn into namedtuples again.
        env_outputs, actor_outputs = batch

        env_outputs = EnvOutput._make(env_outputs)
        actor_outputs = AgentOutput._make(actor_outputs)
        learner_outputs = AgentOutput._make(learner_outputs)

        discounts = (~env_outputs.done).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=actor_outputs.policy_logits,
            target_policy_logits=learner_outputs.policy_logits,
            actions=actor_outputs.action,
            discounts=discounts,
            rewards=env_outputs.reward,
            values=learner_outputs.baseline,
            bootstrap_value=bootstrap_value,
        )

        vtrace_returns = vtrace.VTraceFromLogitsReturns._make(vtrace_returns)

        pg_loss = compute_policy_gradient_loss(
            learner_outputs.policy_logits,
            actor_outputs.action,
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs.baseline
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs.policy_logits
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        total_loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)

        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())

        episode_returns = env_outputs.episode_return[env_outputs.done]

        stats["step"] = stats.get("step", 0) + flags.unroll_length * flags.batch_size
        stats["episode_returns"] = tuple(episode_returns.cpu().numpy())
        stats["mean_environment_return"] = episode_returns.mean().item()
        stats["mean_discriminator_return"] = discriminator_reward.mean().item()
        stats["mean_episode_return"] = (
            stats["mean_environment_return"] + stats["mean_discriminator_return"]
        )
        stats["total_loss"] = total_loss.item()
        stats["pg_loss"] = pg_loss.item()
        stats["baseline_loss"] = baseline_loss.item()
        stats["entropy_loss"] = entropy_loss.item()
        stats["learner_queue_size"] = learner_queue.size()

        if flags.condition and new_frame.size() != 0:
            stats["l2_loss"] = F.mse_loss(
                *new_frame.split(split_size=new_frame.shape[1] // 2, dim=1)
            ).item()

        plogger.log(stats)
        stats["n_discriminator_updates"] = 0
        lock.release()


real_label = 1.0
fake_label = 0.0


def learn_D(
    flags,
    dataloader,
    replay_queue,
    replay_buffer,
    D,
    D_eval,
    optimizer,
    stats,
    plogger,
):
    while True:
        for real, _ in dataloader:
            real = real.to(flags.learner_device, non_blocking=True)

            if flags.condition:
                real = real.repeat(1, 2, 1, 1)

            optimizer.zero_grad()

            p_real = D(real).view(-1)

            label = torch.full(
                (flags.batch_size,), real_label, device=flags.learner_device
            )
            real_loss = F.binary_cross_entropy_with_logits(p_real, label)

            D_x = torch.sigmoid(p_real).mean()

            fake = replay_buffer.sample(flags.batch_size).to(
                flags.learner_device, non_blocking=True
            )

            p_fake = D(fake).view(-1)

            label = torch.full(
                (flags.batch_size,), fake_label, device=flags.learner_device
            )
            fake_loss = F.binary_cross_entropy_with_logits(p_fake, label)

            D_G_z1 = torch.sigmoid(p_fake).mean()

            loss = real_loss + fake_loss

            if flags.gp_weight > 0.0:
                alpha = torch.rand(
                    (flags.batch_size, 1, 1, 1), device=flags.learner_device
                )
                diff = real - fake
                interpolate = fake + alpha * diff
                interpolate.requires_grad = True

                p_interpolate = D(interpolate)

                grad_outputs = torch.ones_like(p_interpolate)

                gradient = torch.autograd.grad(
                    p_interpolate, interpolate, grad_outputs, create_graph=True
                )[0]

                grad_norm = gradient.view(flags.batch_size, -1).norm(2, dim=1)
                grad_penalty = flags.gp_weight * grad_norm.sub(1).pow(2).mean()

                loss = loss + grad_penalty

            loss.backward()

            optimizer.step()

            D_eval.load_state_dict(D.state_dict())

            stats["D_loss"] = loss.item()
            stats["fake_loss"] = fake_loss.item()
            stats["real_loss"] = real_loss.item()
            stats["D_x"] = D_x.item()
            stats["D_G_z1"] = D_G_z1.item()
            stats["n_discriminator_updates"] = (
                stats.get("n_discriminator_updates", 0) + 1
            )
            if flags.gp_weight != 0:
                stats["grad_norm"] = grad_norm.mean().item()
                stats["grad_penalty"] = grad_penalty.item()

            if replay_queue.is_closed():
                return


def train(flags):
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.learner_device = torch.device("cuda")
        flags.actor_device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.learner_device = torch.device("cpu")
        flags.actor_device = torch.device("cpu")

    if flags.max_learner_queue_size is None:
        flags.max_learner_queue_size = flags.batch_size

    # The queue the learner threads will get their data from.
    # Setting `minimum_batch_size == maximum_batch_size`
    # makes the batch size static.
    learner_queue = libtorchbeast.BatchingQueue(
        batch_dim=1,
        minimum_batch_size=flags.batch_size,
        maximum_batch_size=flags.batch_size,
        check_inputs=True,
        maximum_queue_size=flags.max_learner_queue_size,
    )

    # The queue the actorpool stores final render image pairs.
    # A seperate thread will load them to the ReplayBuffer.
    # The batch size of the pairs will be dynamic.
    replay_queue = libtorchbeast.BatchingQueue(
        batch_dim=1,
        minimum_batch_size=1,
        maximum_batch_size=flags.num_actors,
        timeout_ms=100,
        check_inputs=True,
        maximum_queue_size=flags.num_actors,
    )

    if flags.replay_buffer_size is None:
        flags.replay_buffer_size = flags.batch_size * 20

    replay_buffer = ReplayBuffer(flags.replay_buffer_size)

    # The "batcher", a queue for the inference call. Will yield
    # "batch" objects with `get_inputs` and `set_outputs` methods.
    # The batch size of the tensors will be dynamic.
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

    dataset_uses_color = flags.dataset not in ["mnist", "omniglot"]
    grayscale = dataset_uses_color and not flags.use_color
    dataset = utils.create_dataset(flags.dataset, grayscale)

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

    obs_shape = env.observation_space["canvas"].shape
    action_shape = env.action_space.nvec
    order = env.order
    env.close()

    model = models.Net(
        obs_shape=obs_shape,
        order=order,
        action_shape=action_shape,
        grid_shape=(grid_width, grid_width),
    )
    model = model.to(device=flags.learner_device)

    actor_model = models.Net(
        obs_shape=obs_shape,
        order=order,
        action_shape=action_shape,
        grid_shape=(grid_width, grid_width),
    ).eval()
    actor_model.to(device=flags.actor_device)

    if flags.condition:
        D = models.ComplementDiscriminator(obs_shape)
    else:
        D = models.Discriminator(obs_shape)
    D.to(device=flags.learner_device)

    if flags.condition:
        D_eval = models.ComplementDiscriminator(obs_shape)
    else:
        D_eval = models.Discriminator(obs_shape)
    D_eval = D_eval.to(device=flags.learner_device).eval()

    optimizer = optim.Adam(model.parameters(), lr=flags.policy_learning_rate)
    D_optimizer = optim.Adam(
        D.parameters(), lr=flags.discriminator_learning_rate, betas=(0.5, 0.999)
    )

    def lr_lambda(epoch):
        return (
            1
            - min(epoch * flags.unroll_length * flags.batch_size, flags.total_steps)
            / flags.total_steps
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # The ActorPool that will run `flags.num_actors` many loops.
    actors = libtorchbeast.ActorPool(
        unroll_length=flags.unroll_length,
        learner_queue=learner_queue,
        replay_queue=replay_queue,
        inference_batcher=inference_batcher,
        env_server_addresses=addresses,
        initial_agent_state=actor_model.initial_state(),
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

    dataloader = DataLoader(
        dataset,
        batch_size=flags.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    stats = {}

    # Load state from a checkpoint, if possible.
    if os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath, map_location=flags.learner_device
        )
        model.load_state_dict(checkpoint_states["model_state_dict"])
        D.load_state_dict(checkpoint_states["D_state_dict"])
        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
        D_optimizer.load_state_dict(checkpoint_states["D_optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_states["scheduler_state_dict"])
        stats = checkpoint_states["stats"]
        replay_buffer.load_checkpoint(checkpoint_states["replay_buffer"])
        logging.info(f"Resuming preempted job, current stats:\n{stats}")

    # Initialize actor model like learner model.
    actor_model.load_state_dict(model.state_dict())
    D_eval.load_state_dict(D.state_dict())

    learner_threads = [
        threading.Thread(
            target=learn,
            name="learner-thread-%i" % i,
            args=(
                flags,
                learner_queue,
                model,
                actor_model,
                D_eval,
                optimizer,
                scheduler,
                stats,
                plogger,
            ),
        )
        for i in range(flags.num_learner_threads)
    ]

    inference_threads = [
        threading.Thread(
            target=inference,
            name="inference-thread-%i" % i,
            args=(flags, inference_batcher, actor_model,),
        )
        for i in range(flags.num_inference_threads)
    ]

    d_learner = threading.Thread(
        target=learn_D,
        name="d_learner-thread",
        args=(
            flags,
            dataloader,
            replay_queue,
            replay_buffer,
            D,
            D_eval,
            D_optimizer,
            stats,
            plogger,
        ),
    )

    def load_replay_buffer(replay_queue, replay_buffer):
        for obs in replay_queue:
            replay_buffer.push(obs["canvas"].squeeze(0))

    replay_buffer_loader = threading.Thread(
        target=load_replay_buffer,
        name="replay-buffer-loader-thread",
        args=(replay_queue, replay_buffer,),
    )
    replay_buffer_loader.daemon = True

    replay_buffer_loader.start()

    actorpool_thread.start()

    threads = learner_threads + inference_threads

    for t in inference_threads:
        t.start()

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "D_state_dict": D.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "D_optimizer_state_dict": D_optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "stats": stats,
                "flags": vars(flags),
                "replay_buffer": replay_buffer.checkpoint(),
            },
            checkpointpath,
        )

    def format_value(x):
        return f"{x:1.5}" if isinstance(x, float) else str(x)

    try:
        while len(replay_buffer) < flags.batch_size:
            if learner_queue.size() >= flags.batch_size:
                next(learner_queue)
            time.sleep(0.01)

        d_learner.start()

        for t in learner_threads:
            t.start()

        last_checkpoint_time = timeit.default_timer()
        while True:
            start_time = timeit.default_timer()
            start_step = stats.get("step", 0)
            if start_step >= flags.total_steps:
                break
            time.sleep(5)
            end_step = stats.get("step", 0)

            if timeit.default_timer() - last_checkpoint_time > 10 * 60:
                # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timeit.default_timer()

            logging.info(
                "Step %i @ %.1f SPS. Inference batcher size: %i."
                " Learner queue size: %i."
                " Other stats: (%s)",
                end_step,
                (end_step - start_step) / (timeit.default_timer() - start_time),
                inference_batcher.size(),
                learner_queue.size(),
                ", ".join(
                    f"{key} = {format_value(value)}" for key, value in stats.items()
                ),
            )
    except KeyboardInterrupt:
        pass  # Close properly.
    else:
        logging.info("Learning finished after %i steps.", stats["step"])
        checkpoint()

    # Done with learning. Stop all the ongoing work.
    inference_batcher.close()
    learner_queue.close()

    replay_queue.close()

    actorpool_thread.join()

    d_learner.join()

    for t in threads:
        t.join()


def test(flags):
    pass


def main(flags):
    if not flags.pipes_basename.startswith("unix:"):
        raise Exception("--pipes_basename has to be of the form unix:/some/path.")

    if flags.mode == "train":
        if flags.write_profiler_trace:
            logging.info("Running with profiler.")
            with torch.autograd.profiler.profile() as prof:
                train(flags)
            filename = "chrome-%s.trace" % time.strftime("%Y%m%d-%H%M%S")
            logging.info("Writing profiler trace to '%s.gz'", filename)
            prof.export_chrome_trace(filename)
            os.system("gzip %s" % filename)
        else:
            train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
