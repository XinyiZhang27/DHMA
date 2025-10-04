from .distributions import Bernoulli, Categorical, DiagGaussian
import torch
import torch.nn as nn


class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.discrete_action = False

        if action_space.__class__.__name__ == "Discrete":
            self.discrete_action = True
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            for action_dim in action_dims:
                self.action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:  # discrete + continous
            self.mixed_action = True
            self.discrete_dim = action_space[0].n
            self.continuous_dim = action_space[1].shape[0]
            self.action_outs = nn.ModuleList([Categorical(inputs_dim, self.discrete_dim, use_orthogonal, gain),
                                              DiagGaussian(inputs_dim, self.continuous_dim, use_orthogonal, gain)])
    
    def forward(self, x, available_actions=None, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        if self.mixed_action:
            actions = []
            action_log_probs = []
            for i, action_out in enumerate(self.action_outs):
                if available_actions[i] is not None:
                    action_logits = action_out(x, available_actions[i])
                else:
                    action_logits = action_out(x)
                action = action_logits.mode() if deterministic else action_logits.sample()
                action_log_prob = action_logits.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
        elif self.multi_discrete:
            actions = []
            action_log_probs = []
            for i, action_out in enumerate(self.action_outs):
                if available_actions[i] is not None:
                    action_logits = action_out(x, available_actions[i])
                else:
                    action_logits = action_out(x)
                action = action_logits.mode() if deterministic else action_logits.sample()
                action_log_prob = action_logits.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)
        elif self.discrete_action:
            action_logits = self.action_out(x, available_actions)
            actions = action_logits.mode() if deterministic else action_logits.sample()
            action_log_probs = action_logits.log_probs(actions)
        else:
            action_logits = self.action_out(x)
            actions = action_logits.mode() if deterministic else action_logits.sample()
            action_log_probs = action_logits.log_probs(actions)

        return actions, action_log_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if self.mixed_action:
            a, b = action.split((1, self.continuous_dim), -1)
            a = a.long()
            action = [a, b] 
            action_log_probs = [] 
            dist_entropy = []
            for action_out, act, available_action in zip(self.action_outs, action, available_actions):
                if available_action is not None:
                    action_logits = action_out(x, available_action)
                else:
                    action_logits = action_out(x)
                action_log_prob = action_logits.log_probs(act)
                if action_log_prob.ndim == 1:
                    action_log_prob = action_log_prob.unsqueeze(0)
                action_log_probs.append(action_log_prob)
                if active_masks is not None:
                    if len(action_logits.entropy().shape) == len(active_masks.shape):
                        dist_entropy.append((action_logits.entropy() * active_masks).sum()/active_masks.sum())
                    else:
                        dist_entropy.append((action_logits.entropy() * active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logits.entropy().mean())
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = torch.tensor(dist_entropy).mean()

        elif self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act, available_action in zip(self.action_outs, action, available_actions):
                if available_action is not None:
                    action_logits = action_out(x, available_action)
                else:
                    action_logits = action_out(x)
                action_log_prob = action_logits.log_probs(act)
                if action_log_prob.ndim == 1:
                    action_log_prob = action_log_prob.unsqueeze(0)
                action_log_probs.append(action_log_prob)
                if active_masks is not None:
                    dist_entropy.append((action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logits.entropy().mean())
            action_log_probs = torch.cat(action_log_probs, -1)  # ! could be wrong
            dist_entropy = torch.tensor(dist_entropy).mean()

        elif self.discrete_action:
            action_logits = self.action_out(x, available_actions)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()

        else:
            action_logits = self.action_out(x)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                if len(action_logits.entropy().shape) == len(active_masks.shape):
                    dist_entropy = (action_logits.entropy() * active_masks).sum() / active_masks.sum()
                else:
                    dist_entropy = (action_logits.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()

        return action_log_probs, dist_entropy
