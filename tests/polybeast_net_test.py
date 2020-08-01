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
"""Tests for polybeast Net class implementation."""

import unittest

import torch
from torchbeast.core import models


class NetTest(unittest.TestCase):
    def setUp(self):
        self.unroll_length = 4  # Arbitrary.
        self.batch_size = 4  # Arbitrary.
        self.frame_dimension = 64  # Has to match what expected by the model.
        self.action_shape = [1024, 1024, 2, 8]  # First 3 dimensions are fixed.
        self.num_channels = 1  # Has to match with the first conv layer of the net.
        self.grid_shape = [32, 32]  # Specific to each environment.
        self.core_output_size = 256  # Has to match what expected by the model.

        self.obs_shape = [self.num_channels, self.frame_dimension, self.frame_dimension]
        self.inputs = [
            dict(
                canvas=torch.ones(
                    self.unroll_length,
                    self.batch_size,
                    self.num_channels,
                    self.frame_dimension,
                    self.frame_dimension,
                ),
                prev_action=torch.ones(
                    self.unroll_length, self.batch_size, len(self.action_shape)
                ),
                action_mask=torch.ones(
                    self.unroll_length, self.batch_size, len(self.action_shape)
                ),
                noise_sample=torch.ones(self.unroll_length, self.batch_size, 10),
            ),
            torch.zeros(self.batch_size, self.unroll_length, dtype=torch.bool),
        ]

    def test_forward_return_signature(self):
        model = models.Net(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            grid_shape=self.grid_shape,
        )
        core_state = model.initial_state(self.batch_size)

        (action, policy_logits, baseline), core_state = model(*self.inputs, core_state)
        self.assertSequenceEqual(
            action.shape, (self.batch_size, self.unroll_length, len(self.action_shape))
        )
        for logits, num_actions in zip(policy_logits, self.action_shape):
            self.assertSequenceEqual(
                logits.shape, (self.batch_size, self.unroll_length, num_actions)
            )
        self.assertSequenceEqual(baseline.shape, (self.batch_size, self.unroll_length))
        for core_state_element in core_state:
            self.assertSequenceEqual(
                core_state_element.shape, (1, self.batch_size, self.core_output_size),
            )

    def test_initial_state(self):
        model = models.Net(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            grid_shape=self.grid_shape,
        )
        core_state = model.initial_state(self.batch_size)

        self.assertEqual(len(core_state), 2)
        for core_state_element in core_state:
            self.assertSequenceEqual(
                core_state_element.shape, (1, self.batch_size, self.core_output_size),
            )


if __name__ == "__main__":
    unittest.main()
