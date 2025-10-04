import gym
from gym import spaces
import numpy as np
from envs.env_core import EnvCore


class Env(object):
    """
    对于环境的封装
    """

    def __init__(self, all_args, Edge_IDs, edge_index_map, uav_edge_map, shortest_paths, num_uav_cover, time_slot):
        self.env = EnvCore(all_args, Edge_IDs, edge_index_map, uav_edge_map, shortest_paths, num_uav_cover, time_slot)
        self.num_edge_agent = self.env.num_edge
        self.num_uav_agent = self.env.num_uav

        self.mappo_uav_obs_dim = self.env.obs_uav_dim
        self.mappo_uav_action_dim = self.env.action_uav_dim
        self.mappo_uav_share_obs_dim = self.env.share_obs_uav_dim
        self.mappo_edge_obs_dim = self.env.obs_edge_dim
        self.mappo_edge_discrete_a_dim = self.env.action_discrete_dim
        self.mappo_edge_continuous_a_dim = self.env.action_continuous_dim
        self.mappo_edge_share_obs_dim = self.env.share_obs_edge_dim

        # configure spaces
        self.mappo_uav_obs_space = []
        self.mappo_uav_action_space = []
        self.mappo_uav_share_obs_space = []
        self.mappo_edge_obs_space = []
        self.mappo_edge_action_space = []
        self.mappo_edge_share_obs_space = []

        for agent_idx in range(self.num_uav_agent):
            self.mappo_uav_action_space.append(spaces.Discrete(self.mappo_uav_action_dim[agent_idx]))
            self.mappo_uav_obs_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.mappo_uav_obs_dim[agent_idx],), dtype=np.float32))
            self.mappo_uav_share_obs_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.mappo_uav_share_obs_dim,), dtype=np.float32))
        for agent_idx in range(self.num_edge_agent):
            self.mappo_edge_action_space.append(spaces.Tuple((spaces.Discrete(self.mappo_edge_discrete_a_dim),
                                                             spaces.Box(low=-np.inf, high=+np.inf, shape=(self.mappo_edge_continuous_a_dim,), dtype=np.float32))))
            self.mappo_edge_obs_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.mappo_edge_obs_dim[agent_idx],), dtype=np.float32))
            self.mappo_edge_share_obs_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.mappo_edge_share_obs_dim[agent_idx],), dtype=np.float32))

    def step(self, edge_actions, edge_vehicle_map, next_edge_vehicle_map, future_trajectory):
        all_task_num, edge_task_nums, edge_comp_task_nums, uav_comp_task_nums, uav_obs, uav_share_obs, uav_rewards,\
        edge_obs, edge_share_obs, edge_next_obs, edge_next_share_obs, edge_rewards, info, done = self.env.step(
            edge_actions, edge_vehicle_map, next_edge_vehicle_map, future_trajectory)
        return all_task_num, edge_task_nums, edge_comp_task_nums, uav_comp_task_nums, \
               uav_obs, uav_share_obs, uav_rewards, \
               edge_obs, edge_share_obs, edge_next_obs, edge_next_share_obs, edge_rewards, info, done

    def reset(self, edge_vehicle_map, future_trajectory):
        all_task_num, edge_task_nums, uav_obs, uav_share_obs, edge_obs, edge_share_obs = self.env.reset(edge_vehicle_map, future_trajectory)
        return all_task_num, edge_task_nums, uav_obs, uav_share_obs, edge_obs, edge_share_obs

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass
