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

import torch
import numpy as np
import gym
import cv2

cv2.ocl.setUseOpenCL(False)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=64, height=64):
        super().__init__(env)
        self._width = width
        self._height = height

        original_space = env.observation_space
        c = original_space.shape[-1]
        new_space = gym.spaces.Box(
            low=0, high=255, shape=(self._height, self._width, c), dtype=np.uint8,
        )
        self.observation_space = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

        if c == 1:
            self._grayscale = True
        else:
            self._grayscale = False

    def observation(self, obs):
        frame = cv2.resize(
            obs["canvas"], (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)
        obs["canvas"] = frame

        return obs


class Base(gym.Wrapper):
    def __init__(self, env, noise_dim=10):
        super(Base, self).__init__(env)
        self.dim = noise_dim
        self._initial_action = np.zeros(env.action_space.nvec.shape, dtype=np.int64)

        original_space = env.observation_space
        h, w, c = original_space.shape
        new_space = gym.spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8,)
        self.observation_space = new_space

    def _convert_to_dict(self, action):
        return dict(zip(self.env.order, action.squeeze().tolist()))

    def _to_NCHW(self, canvas):
        return np.transpose(canvas, axes=(2, 0, 1))

    def _sample_noise(self, dims):
        return np.random.normal(size=(dims,)).astype(np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(self._convert_to_dict(action))
        obs["canvas"] = self._to_NCHW(obs["canvas"])
        obs["noise_sample"] = self.noise
        obs["prev_action"] = self.prev_action
        self.prev_action = action
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs["canvas"] = self._to_NCHW(obs["canvas"])

        self.noise = self._sample_noise(self.dim)
        obs["noise_sample"] = self.noise

        self.prev_action = self._initial_action
        obs["prev_action"] = self.prev_action

        return obs


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
        super(ConcatTarget, self).__init__(env)
        self.data = iter(dataset)

    def _concat(self, canvas):
        return torch.cat([canvas, self.target])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs["canvas"] = self._concat(obs["canvas"])
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs["canvas"] = self._concat(obs["canvas"])
        return obs
