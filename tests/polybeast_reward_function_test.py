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
"""Tests for polybeast learn function implementation."""

import unittest
from unittest import mock

import torch
from torchbeast import polybeast_learner as polybeast
from torchbeast.core import models


def _state_dict_to_numpy(state_dict):
    return {key: value.numpy() for key, value in state_dict.items()}


class RewardTest(unittest.TestCase):
    def setUp(self):
        unroll_length = 3  # Inference called for every step.
        batch_size = 4  # Arbitrary.
        frame_dimension = 64  # Has to match what expected by the model.
        num_channels = 1  # Has to match with the first conv layer of the net.

        obs_shape = [num_channels, frame_dimension, frame_dimension]

        # Set the random seed manually to get reproducible results.
        torch.manual_seed(0)

        self.D = models.Discriminator(obs_shape).eval()

        # Mock flags.
        mock_flags = mock.Mock()
        mock_flags.learner_device = torch.device("cpu")
        mock_flags.unroll_length = unroll_length - 1
        mock_flags.batch_size = batch_size
        self.flags = mock_flags

        # Prepare content for mock_learner_queue.
        self.obs = dict(canvas=torch.ones([unroll_length, batch_size] + obs_shape))
        self.done = torch.tensor(
            [
                [True, False, False, True],
                [False, True, False, False],
                [True, False, True, False],
            ]
        )

        self.new_frame = torch.ones([unroll_length - 1, batch_size] + obs_shape)

        self.reward = torch.randn(len(self.done[1:].nonzero(as_tuple=False)))

    def test_tca_reward_shape(self):
        """Check that the tca reward shape is correct."""
        reward = polybeast.tca_reward_function(
            self.flags, self.obs, self.new_frame, self.D
        )

        self.assertEqual(
            list(reward.shape), [self.flags.unroll_length + 1, self.flags.batch_size]
        )

    def test_reward_shape(self):
        """Check that the reward shape is correct."""
        reward = polybeast.reward_function(
            self.flags, self.done, self.new_frame, self.D
        )

        self.assertEqual(
            list(reward.nonzero(as_tuple=False).shape),
            list(self.done[1:].nonzero(as_tuple=False).shape),
        )

    def test_non_zero_tca_reward(self):
        """Check that the tca reward is not zero."""
        reward = polybeast.tca_reward_function(
            self.flags, self.obs, self.new_frame, self.D.train()
        )

        self.assertNotEqual(reward.sum().item(), 0.0)

    def test_non_zero_reward(self):
        """Check that the reward is not zero."""
        reward = polybeast.reward_function(
            self.flags, self.done, self.new_frame, self.D
        )

        self.assertNotEqual(reward.sum().item(), 0.0)

    def test_tca_reward_order(self):
        """Check that the tca reward order is correct."""
        reward = polybeast.tca_reward_function(
            self.flags, self.obs, self.new_frame, self.D
        )

        frame = self.obs["canvas"][:-1]
        for i in range(self.flags.unroll_length):
            self.assertEqual(
                reward[i + 1].sum().item(),
                (self.D(self.new_frame[i]) - self.D(frame[i])).sum().item(),
            )

    def test_reward_order(self):
        """Check that the reward order is correct."""

        def mock_D(new_frame):
            return self.reward

        reward = polybeast.reward_function(
            self.flags, self.done, self.new_frame, mock_D
        )
        index = reward.nonzero(as_tuple=False)
        for i, index in enumerate(index.split(1)):
            index = index.squeeze()
            self.assertEqual(reward[index[0], index[1]].item(), self.reward[i].item())


if __name__ == "__main__":
    unittest.main()
