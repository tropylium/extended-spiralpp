# The MIT License
#
# Copyright (c) 2017 OpenAI (http://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in # all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Taken from
#   https://raw.githubusercontent.com/openai/baselines/7c520852d9cf4eaaad326a3d548efc915dc60c10/baselines/common/atari_wrappers.py
# and slightly modified by (c) Facebook, Inc. and its affiliates.
# 2 May 2020 Modified by urw7rs

import numpy as np
import gym
from gym import spaces
import cv2

cv2.ocl.setUseOpenCL(False)

from torch.utils.data import DataLoader


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=64, height=64, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._expand_dims = grayscale
        self._key = dict_space_key

        if self._expand_dims:
            num_colors = 1
        else:
            num_colors = 3

        if self._key is None:
            original_space = self.observation_space
        else:
            original_space = self.observation_space.spaces[self._key]

        self._grayscale = original_space.shape[-1] == 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            self.observation_space = new_space
        else:
            self.observation_space.spaces[self._key] = new_space

        assert original_space.dtype == np.uint8 and len(original_space.shape) in [1, 3]

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._expand_dims:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FloatNCHW(gym.ObservationWrapper):
    def __init__(self, env, dict_space_key=None):
        super().__init__(env)
        self._key = dict_space_key

        if self._key is None:
            original_space = self.observation_space
        else:
            original_space = self.observation_space.spaces[self._key]

        h, w, c = original_space.shape
        new_space = spaces.Box(low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32)

        self.observation_space.spaces[dict_space_key] = new_space
        assert len(original_space.shape) in [1, 3]

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        nchw = np.transpose(frame, axes=(2, 0, 1)) / 255.0

        if self._key is None:
            obs = nchw
        else:
            obs = obs.copy()
            obs[self._key] = nchw
        return obs


class SampleNoise(gym.Wrapper):
    def __init__(
        self, env, dict_space_key, noise_dim=10,
    ):
        super().__init__(env)
        self.dim = noise_dim
        self._key = dict_space_key

        new_space = env.observation_space.spaces
        new_space["noise_sample"] = spaces.Box(
            low=0.0, high=1.0, shape=(noise_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(new_space)

    def reset(self):
        obs = self.env.reset()
        self.noise = np.random.normal(size=(self.dim,)).astype(np.float32)
        obs[self._key] = self.noise
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs[self._key] = self.noise
        return obs, reward, done, info


class SavePrevAction(gym.Wrapper):
    def __init__(self, env, dict_space_key):
        super().__init__(env)

        self._key = dict_space_key

        if isinstance(self.action_space, spaces.MultiDiscrete):
            self.action_space = spaces.MultiDiscrete(self.action_space.nvec)
            self._initial_action = np.zeros(
                self.action_space.nvec.shape, dtype=np.int64
            )

    def reset(self):
        obs = self.env.reset()
        obs[self._key] = self._initial_action
        self.prev_action = self._initial_action

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs[self._key] = self.prev_action
        self.prev_action = action

        return obs, reward, done, info


def make_raw(env_id, config):
    env = gym.make("spiral:" + env_id)
    if len(config) > 0:
        env.configure(**config)
    return env


class ConcatTarget(gym.Wrapper):
    """
    Concat target to obs
    """

    def __init__(self, env, dataset):
        super().__init__(env)
        self.dataloader = DataLoader(dataset, shuffle=True, pin_memory=True)
        self.iterator = iter(self.dataloader)

        new_space = env.observation_space.spaces

        c, h, w = new_space["canvas"].shape
        new_space["canvas"] = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(c * 2, h, w), dtype=np.float32,
        )
        self.observation_space = spaces.Dict(new_space)

    def _concat(self, canvas):
        return np.concatenate([canvas, self.target])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = obs.copy()
        obs["canvas"] = self._concat(obs["canvas"])
        return obs, reward, done, info

    def reset(self):
        try:
            target, _ = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            target, _ = next(self.iterator)

        self.target = target.squeeze(0).numpy()

        obs = self.env.reset()
        obs = obs.copy()
        obs["canvas"] = self._concat(obs["canvas"])

        return obs
