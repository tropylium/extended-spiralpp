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

import copy
import unittest
from unittest import mock

import numpy as np
import torch
from torchbeast import polybeast_learner as polybeast
from torchbeast.core import models


def _state_dict_to_numpy(state_dict):
    return {key: value.numpy() for key, value in state_dict.items()}


class LearnTest(unittest.TestCase):
    def setUp(self):
        unroll_length = 2  # Inference called for every step.
        batch_size = 4  # Arbitrary.
        frame_dimension = 64  # Has to match what expected by the model.
        action_shape = [1024, 1024, 2, 8, 10]
        num_channels = 1  # Has to match with the first conv layer of the net.
        grid_shape = [32, 32]  # Specific to each environment.

        obs_shape = [num_channels, frame_dimension, frame_dimension]

        # The following hyperparamaters are arbitrary.
        self.lr = 0.1
        total_steps = 100000

        # Set the random seed manually to get reproducible results.
        torch.manual_seed(0)

        self.model = models.Net(
            obs_shape=obs_shape, action_shape=action_shape, grid_shape=grid_shape,
        )
        self.actor_model = models.Net(
            obs_shape=obs_shape, action_shape=action_shape, grid_shape=grid_shape,
        )
        self.initial_model_dict = copy.deepcopy(self.model.state_dict())
        self.initial_actor_model_dict = copy.deepcopy(self.actor_model.state_dict())

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        self.D = models.Discriminator(obs_shape)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=total_steps // 10
        )

        self.stats = {}

        # The call to plogger.log will not perform any action.
        plogger = mock.Mock()
        plogger.log = mock.Mock()

        # Mock flags.
        mock_flags = mock.Mock()
        mock_flags.learner_device = torch.device("cpu")
        mock_flags.reward_clipping = "abs_one"  # Default value from cmd.
        mock_flags.discounting = 0.99  # Default value from cmd.
        mock_flags.baseline_cost = 0.5  # Default value from cmd.
        mock_flags.entropy_cost = 0.0006  # Default value from cmd.
        mock_flags.unroll_length = unroll_length
        mock_flags.batch_size = batch_size
        mock_flags.grad_norm_clipping = 40
        mock_flags.use_tca = True
        mock_flags.condition = False

        # Prepare content for mock_learner_queue.
        obs = dict(
            canvas=torch.ones(
                unroll_length,
                batch_size,
                num_channels,
                frame_dimension,
                frame_dimension,
            ),
            prev_action=torch.ones(unroll_length, batch_size, len(action_shape)),
            action_mask=torch.ones(unroll_length, batch_size, len(action_shape)),
            noise_sample=torch.ones(unroll_length, batch_size, 10),
        )
        rewards = torch.ones(unroll_length, batch_size)
        done = torch.zeros(unroll_length, batch_size, dtype=torch.bool)
        episode_step = torch.ones(unroll_length, batch_size)
        episode_return = torch.ones(unroll_length, batch_size)

        new_frame = dict(
            canvas=torch.ones(
                1, batch_size, num_channels, frame_dimension, frame_dimension,
            ),
            prev_action=torch.ones(1, batch_size, len(action_shape)),
            action_mask=torch.ones(1, batch_size, len(action_shape)),
            noise_sample=torch.ones(1, batch_size, 10),
        )

        env_outputs = (obs, rewards, done, episode_step, episode_return)
        actor_outputs = (
            # Actions taken.
            torch.cat(
                list(
                    map(
                        lambda num_actions: torch.randint(
                            low=0, high=num_actions, size=(unroll_length, batch_size, 1)
                        ),
                        action_shape,
                    )
                ),
                dim=-1,
            ),
            # Logits.
            list(
                map(
                    lambda num_actions: torch.randn(
                        unroll_length, batch_size, num_actions
                    ),
                    action_shape,
                )
            ),
            # Baseline.
            torch.rand(unroll_length, batch_size),
        )
        initial_agent_state = self.model.initial_state(batch_size)
        tensors = ((env_outputs, actor_outputs), new_frame, initial_agent_state)

        # Mock learner_queue.
        mock_learner_queue = mock.MagicMock()
        mock_learner_queue.__iter__.return_value = iter([tensors])

        self.learn_args = (
            mock_flags,
            mock_learner_queue,
            self.model,
            self.actor_model,
            self.D,
            optimizer,
            scheduler,
            self.stats,
            plogger,
        )

    def test_parameters_copied_to_actor_model(self):
        """Check that the learner model copies the parameters to the actor model."""
        # Reset models.
        self.model.load_state_dict(self.initial_model_dict)
        self.actor_model.load_state_dict(self.initial_actor_model_dict)

        polybeast.learn(*self.learn_args)

        np.testing.assert_equal(
            _state_dict_to_numpy(self.actor_model.state_dict()),
            _state_dict_to_numpy(self.model.state_dict()),
        )

    def test_weights_update(self):
        """Check that trainable parameters get updated after one iteration."""
        # Reset models.
        self.model.load_state_dict(self.initial_model_dict)
        self.actor_model.load_state_dict(self.initial_actor_model_dict)

        polybeast.learn(*self.learn_args)

        model_state_dict = self.model.state_dict(keep_vars=True)
        actor_model_state_dict = self.actor_model.state_dict(keep_vars=True)
        for key, initial_tensor in self.initial_model_dict.items():
            model_tensor = model_state_dict[key]
            actor_model_tensor = actor_model_state_dict[key]
            # Assert that the gradient is not zero for the learner.
            if model_tensor.grad is None or torch.norm(model_tensor.grad) == 0:
                print(key)
                continue
            self.assertGreater(torch.norm(model_tensor.grad), 0.0)
            # Assert actor has no gradient.
            # Note that even though actor model tensors have no gradient,
            # they have requires_grad == True. No gradients are ever calculated
            # for these tensors because the inference function in polybeast.py
            # (that performs forward passes with the actor_model) uses torch.no_grad
            # context manager.
            self.assertIsNone(actor_model_tensor.grad)
            # Assert that the weights are updated in the expected way.
            # We manually perform a gradient descent step,
            # and check that they are the same as the calculated ones
            # (ignoring floating point errors).
            expected_tensor = (
                initial_tensor.detach().numpy() - self.lr * model_tensor.grad.numpy()
            )
            np.testing.assert_almost_equal(
                model_tensor.detach().numpy(), expected_tensor
            )
            np.testing.assert_almost_equal(
                actor_model_tensor.detach().numpy(), expected_tensor
            )

    def test_gradients_update(self):
        """Check that gradients get updated after one iteration."""
        # Reset models.
        self.model.load_state_dict(self.initial_model_dict)
        self.actor_model.load_state_dict(self.initial_actor_model_dict)

        # There should be no calculated gradient yet.
        for p in self.model.parameters():
            self.assertIsNone(p.grad)
        for p in self.actor_model.parameters():
            self.assertIsNone(p.grad)

        polybeast.learn(*self.learn_args)

        # Check that every parameter for the learner model has a gradient, and that
        # there is at least some non-zero gradient for each set of paramaters.
        for p in self.model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertFalse(torch.equal(p.grad, torch.zeros_like(p.grad)))

        # Check that the actor model has no gradients associated with it.
        for p in self.actor_model.parameters():
            self.assertIsNone(p.grad)

    def test_non_zero_loss(self):
        """Check that the loss is not zero after one iteration."""
        # Reset models.
        self.model.load_state_dict(self.initial_model_dict)
        self.actor_model.load_state_dict(self.initial_actor_model_dict)

        polybeast.learn(*self.learn_args)

        self.assertNotEqual(self.stats["total_loss"], 0.0)
        self.assertNotEqual(self.stats["pg_loss"], 0.0)
        self.assertNotEqual(self.stats["baseline_loss"], 0.0)
        self.assertNotEqual(self.stats["entropy_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
