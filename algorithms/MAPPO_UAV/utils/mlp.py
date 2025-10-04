import torch.nn as nn
from algorithms.util import init, get_clones

"""MLP modules."""

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        if self._layer_N == 1:
            self.fc2 = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))

    def forward(self, x):
        x = self.fc1(x)
        if self._layer_N == 1:
            x = self.fc2(x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization0
        self._use_orthogonal = args.use_orthogonal0
        self._use_ReLU = args.use_ReLU0
        self._stacked_frames = args.stacked_frames0
        self._layer_N = args.layer_N0
        self.hidden_size = args.hidden_size0

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x
