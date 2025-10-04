import torch
import numpy as np

from myutils.util import check, get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1,0,2).reshape(-1, *x.shape[2:])


class SeparatedReplayBuffer(object):
    def __init__(self, args, obs_space, share_obs_space, act_space):
        self.buffer_length = args.buffer_length1
        self.rnn_hidden_size = args.hidden_size1
        self.recurrent_N = args.recurrent_N1
        self.gamma = args.gamma1
        self.gae_lambda = args.gae_lambda1
        self._use_gae = args.use_gae1
        self._use_popart = args.use_popart1
        self._use_valuenorm = args.use_valuenorm1
        self._use_proper_time_limits = args.use_proper_time_limits1

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros((self.buffer_length, *share_obs_shape), dtype=np.float32)
        self.obs = np.zeros((self.buffer_length, *obs_shape), dtype=np.float32)
        self.next_share_obs = np.zeros((self.buffer_length, *share_obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((self.buffer_length, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros((self.buffer_length, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros((self.buffer_length, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_length, 1), dtype=np.float32)
        
        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.buffer_length, act_space.n), dtype=np.float32)
        elif act_space.__class__.__name__ == 'Tuple':
            self.available_actions = []
            for subspace in act_space:
                if subspace.__class__.__name__ == 'Discrete':
                    self.available_actions.append(np.ones((self.buffer_length, subspace.n), dtype=np.float32))
                else:
                    self.available_actions.append(None)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros((self.buffer_length, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros((self.buffer_length, act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_length, 1), dtype=np.float32)
        
        self.masks = np.ones((self.buffer_length, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def after_update(self):
        self.share_obs.fill(0)
        self.obs.fill(0)
        self.next_share_obs.fill(0)
        self.next_obs.fill(0)

        self.rnn_states[0] = self.rnn_states[self.step].copy()
        self.rnn_states[1:] = 0
        self.rnn_states_critic[0] = self.rnn_states_critic[self.step].copy()
        self.rnn_states_critic[1:] = 0

        self.value_preds.fill(0)
        self.returns.fill(0)
        self.actions.fill(0)
        self.action_log_probs.fill(0)
        self.rewards.fill(0)

        self.masks.fill(1)
        self.bad_masks.fill(1)
        self.active_masks.fill(1)
        if self.available_actions is not None:
            if isinstance(self.available_actions, list):
                for i, available_action in enumerate(self.available_actions):
                    if available_action is not None:
                        self.available_actions[i].fill(1)
            else:
                self.available_actions.fill(1)
        self.step = 0

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_proper_time_limits:  # default=False
            if self._use_gae:
                self.value_preds[self.step] = next_value
                gae = 0
                for step in reversed(range(self.step)):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[
                            step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[self.step] = next_value
                for step in reversed(range(self.step)):
                    if self._use_popart:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                            + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                            + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:  # default=True
                self.value_preds[self.step] = next_value
                gae = 0
                for step in reversed(range(self.step)):
                    if self._use_popart or self._use_valuenorm:  # default=False, True
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[self.step] = next_value
                for step in reversed(range(self.step)):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        batch_size = self.step

        if mini_batch_size is None:
            """
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of steps ({})"
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(batch_size, num_mini_batch))
            """
            if batch_size >= num_mini_batch:
                mini_batch_size = batch_size // num_mini_batch
            else:
                mini_batch_size = batch_size

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size]
                   for i in range(num_mini_batch if batch_size >= num_mini_batch else 0)]

        share_obs = self.share_obs[:self.step].reshape(-1, *self.share_obs.shape[1:])
        obs = self.obs[:self.step].reshape(-1, *self.obs.shape[1:])
        rnn_states = self.rnn_states[:self.step].reshape(-1, *self.rnn_states.shape[1:])
        rnn_states_critic = self.rnn_states_critic[:self.step].reshape(-1, *self.rnn_states_critic.shape[1:])
        actions = self.actions[:self.step].reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            if isinstance(self.available_actions, list):
                available_actions = [available_action[:self.step].reshape(-1, available_action.shape[-1]) if available_action is not None else None
                                     for available_action in self.available_actions]
            else:
                available_actions = self.available_actions[:self.step].reshape(-1, self.available_actions.shape[-1])

        value_preds = self.value_preds[:self.step].reshape(-1, 1)
        returns = self.returns[:self.step].reshape(-1, 1)
        masks = self.masks[:self.step].reshape(-1, 1)
        active_masks = self.active_masks[:self.step].reshape(-1, 1)
        action_log_probs = self.action_log_probs[:self.step].reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                if isinstance(self.available_actions, list):
                    available_actions_batch = [available_action[indices] if available_action is not None else None
                                               for available_action in available_actions]
                else:
                    available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
                  adv_targ, available_actions_batch
