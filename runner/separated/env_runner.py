import time
import os
import numpy as np
import pandas as pd
from itertools import chain
from myutils.util import check_and_create_path, load_edge_ids, load_pkl
import torch
from runner.separated.base_runner import Runner
from runner.separated.trajectory_prediction import InformerModule, Trajectory_Loader


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
        self.simulation_scenario = self.all_args.simulation_scenario
        self.time_range = self.all_args.time_range
        self.time_slots = self.all_args.time_slot  # 车辆移动过程中进行60个时间片(s)的任务，每个时间片一个任务

        # 加载server坐标
        edge_pos_file = "sumo/data/{}/edge_position.csv".format(self.simulation_scenario)
        self.edge_pos = pd.read_csv(edge_pos_file)

        # 轨迹预测模型
        self.seq_len = 32
        self.label_len = 16
        self.pred_len = 8
        self.TrajPredictionModel = InformerModule(enc_in=2, dec_in=2, c_out=2, seq_len=self.seq_len, label_len=self.label_len,
                                   pred_len=self.pred_len, factor=5, d_model=512, n_heads=8, e_layers=2, d_layers=1,
                                   d_ff=2048, dropout=0.05, attn='prob', embed='timeF', freq='s', activation='gelu',
                                   output_attention=False, distil=True, mix=True, device=torch.device('cuda:0'))
        setting = '{}_{}/seq{}label{}pre{}itr{}'.format(
            self.simulation_scenario, self.time_range, self.seq_len, self.label_len, self.pred_len, 0)
        self.TrajPredictionModel.load(setting)  # load model and scaler

        # 加载模拟数据 (初步使用模拟和预处理好的数据, 后续可改为实时调用的API)
        simulation_data_file = "sumo/data/{}/{}/simulation_sequences.pkl".format(self.simulation_scenario, self.time_range)
        # simulation_sequences --- key: start_time, value: one_sequence
        # one_sequence --- time_len=60长度的edge_vehicle_map --- key: edge_id, value: list of vehicle_id
        # {start_time: [{edge_0:[v_1,v_2,...., ],edge_1:[],...,edge_num:[]},{},...,time_len],...,episode}
        self.simulation_sequences = load_pkl(simulation_data_file)
        self.sequence_keys = list(self.simulation_sequences.keys())  # list of start_time
        self.episode = len(self.sequence_keys)  # 迭代次数
        print("training episode: {}".format(self.episode))
        print(self.sequence_keys)

        # 加载车辆历史轨迹数据
        vehicle_trajectory_file = "st_prediction/trajectory_data/{}/{}/vehicle_trajectory.pkl".format(
            self.simulation_scenario, self.time_range)
        self.vehicle_trajectory = load_pkl(vehicle_trajectory_file)

        # 加载每个时刻系统内车辆数据
        time_vehicle_file = "st_prediction/trajectory_data/{}/{}/time_vehicle.pkl".format(
            self.simulation_scenario, self.time_range)
        self.time_vehicle = load_pkl(time_vehicle_file)

    def get_prediction_traj(self, start_time):
        # 进行预测
        current_vehicles = self.time_vehicle[start_time]  # start_time时刻系统内的车辆
        future_trajectory = {}  # key: vehicle_id, value: predict_trajectory (numpy)
        predict_vehicles = []  # 记录放入history_trajectories中的车的顺序
        seqs_x, seqs_x_mark, seqs_y, seqs_y_mark = [], [], [], []
        for vehicle in current_vehicles:
            time_and_trajectory = self.vehicle_trajectory[vehicle]
            trajectory_loader = Trajectory_Loader(time_and_trajectory)
            query_flag, seq_x, seq_x_mark, seq_y, seq_y_mark = trajectory_loader.query_history(
                start_time, self.seq_len, self.label_len, self.pred_len)
            if query_flag == 1:
                predict_vehicles.append(vehicle)
                seqs_x.append(seq_x)
                seqs_x_mark.append(seq_x_mark)
                seqs_y.append(seq_y)
                seqs_y_mark.append(seq_y_mark)
            else:
                future_trajectory[vehicle] = None
        if len(predict_vehicles) > 0:
            predict_edge_ids = self.TrajPredictionModel.predict(self.edge_pos,
                               np.array(seqs_x), np.array(seqs_x_mark), np.array(seqs_y), np.array(seqs_y_mark))
            for index, vehicle in enumerate(predict_vehicles):
                future_trajectory[vehicle] = predict_edge_ids[index]
        return future_trajectory

    @torch.no_grad()
    def collect_uav(self, uav_obs, uav_share_obs, uav_decision_counts):
        uav_actions = {}
        uav_actions_env = {}
        for uav_agent_id in range(self.num_uav_agents):
            self.mappo_uav_trainer[uav_agent_id].prep_rollout()
            uav_agent_obs = uav_obs[uav_agent_id]
            uav_agent_share_obs = uav_share_obs[uav_agent_id]
            if not self.use_centralized_Q0:  # default: True
                uav_agent_share_obs = uav_obs[uav_agent_id]
            step = self.mappo_uav_buffer[uav_agent_id].step
            self.mappo_uav_buffer[uav_agent_id].obs[step] = uav_agent_obs.copy()
            self.mappo_uav_buffer[uav_agent_id].share_obs[step] = uav_agent_share_obs.copy()
            value, uav_action, action_log_prob, rnn_state, rnn_state_critic = self.mappo_uav_trainer[
                uav_agent_id].policy.get_actions(
                self.mappo_uav_buffer[uav_agent_id].share_obs[step],
                self.mappo_uav_buffer[uav_agent_id].obs[step],
                self.mappo_uav_buffer[uav_agent_id].rnn_states[step],
                self.mappo_uav_buffer[uav_agent_id].rnn_states_critic[step],
                self.mappo_uav_buffer[uav_agent_id].masks[step],
            )
            self.mappo_uav_buffer[uav_agent_id].value_preds[step] = _t2n(value).copy()
            self.mappo_uav_buffer[uav_agent_id].actions[step] = _t2n(uav_action).copy()
            self.mappo_uav_buffer[uav_agent_id].action_log_probs[step] = _t2n(action_log_prob).copy()
            self.mappo_uav_buffer[uav_agent_id].rnn_states[step + 1] = np.squeeze(_t2n(rnn_state), axis=0).copy()
            self.mappo_uav_buffer[uav_agent_id].rnn_states_critic[step + 1] = np.squeeze(_t2n(rnn_state_critic), axis=0).copy()

            self.mappo_uav_buffer[uav_agent_id].step = (step + 1) % self.mappo_uav_buffer[uav_agent_id].buffer_length

            uav_action = _t2n(uav_action)[0]
            uav_action_env = np.eye(self.envs.mappo_uav_action_space[uav_agent_id].n)[uav_action]  # rearrange action
            for agent_id in self.uav_edge_map[uav_agent_id]:
                uav_actions[agent_id] = uav_action
                uav_actions_env[agent_id] = uav_action_env
            uav_decision_counts[uav_agent_id][uav_action] += 1
        return uav_actions, uav_actions_env, uav_decision_counts

    @torch.no_grad()
    def collect_edge(self, edge_obs, edge_share_obs, uav_actions_env):
        edge_actions = {}
        for agent_id in range(self.num_edge_agents):
            self.mappo_edge_trainer[agent_id].prep_rollout()
            if agent_id in edge_obs.keys():
                agent_off_obs = edge_obs[agent_id]  # (vehicle num, )
                agent_off_share_obs = edge_share_obs[agent_id]    # (vehicle num, )
                if not self.use_centralized_Q1:  # default: True
                    agent_off_share_obs = edge_obs[agent_id]

                flag_uav_available = True
                # 判断该edge被哪个uav所覆盖 uav_id
                for uav_id, covered_edges in self.uav_edge_map.items():
                    if agent_id in covered_edges:
                        break
                if self.uav_edge_map[uav_id][np.argmax(uav_actions_env[agent_id])] != agent_id:
                    flag_uav_available = False

                agent_edge_actions = []
                for agent_off_ob, agent_off_share_ob in zip(agent_off_obs, agent_off_share_obs):
                    step = self.mappo_edge_buffer[agent_id].step
                    self.mappo_edge_buffer[agent_id].obs[step] = np.hstack((agent_off_ob.copy(), uav_actions_env[agent_id]))
                    self.mappo_edge_buffer[agent_id].share_obs[step] = np.hstack((agent_off_share_ob.copy(), uav_actions_env[agent_id]))
                    if flag_uav_available == False:
                        self.mappo_edge_buffer[agent_id].available_actions[0][step][-1] = 0
                    value, edge_action, action_log_prob, rnn_state, rnn_state_critic = self.mappo_edge_trainer[agent_id].policy.get_actions(
                        self.mappo_edge_buffer[agent_id].share_obs[step],
                        self.mappo_edge_buffer[agent_id].obs[step],
                        self.mappo_edge_buffer[agent_id].rnn_states[step],
                        self.mappo_edge_buffer[agent_id].rnn_states_critic[step],
                        self.mappo_edge_buffer[agent_id].masks[step],
                        [self.mappo_edge_buffer[agent_id].available_actions[0][step], None]
                    )
                    self.mappo_edge_buffer[agent_id].value_preds[step] = _t2n(value).copy()
                    self.mappo_edge_buffer[agent_id].actions[step] = _t2n(edge_action).copy()
                    self.mappo_edge_buffer[agent_id].action_log_probs[step] = _t2n(action_log_prob).copy()
                    self.mappo_edge_buffer[agent_id].rnn_states[step + 1] = np.squeeze(_t2n(rnn_state), axis=0).copy()
                    self.mappo_edge_buffer[agent_id].rnn_states_critic[step + 1] = np.squeeze(_t2n(rnn_state_critic), axis=0).copy()

                    self.mappo_edge_buffer[agent_id].step = (step + 1) % self.mappo_edge_buffer[agent_id].buffer_length

                    edge_action = _t2n(edge_action)
                    agent_edge_actions.append(edge_action)

                edge_actions[agent_id] = np.array(agent_edge_actions)
        return edge_actions

    def insert_mappo_uav(self, counters, next_obs, next_share_obs, rewards):
        for agent_id in range(self.num_uav_agents):
            if not self.use_centralized_Q0:  # default: False - independent
                next_share_obs[agent_id] = next_obs[agent_id]
            step = counters[agent_id]
            self.mappo_uav_buffer[agent_id].rewards[step] = rewards[agent_id]
            self.mappo_uav_buffer[agent_id].next_obs[step] = next_obs[agent_id].copy()
            self.mappo_uav_buffer[agent_id].next_share_obs[step] = next_share_obs[agent_id].copy()
            counters[agent_id] = (step + 1) % self.mappo_uav_buffer[agent_id].buffer_length

    def insert_mappo_edge(self, counters, next_obs, next_share_obs, rewards, uav_actions_env):
        for agent_id in range(self.num_edge_agents):
            if agent_id in next_obs.keys():
                agent_next_obs = next_obs[agent_id]
                agent_next_share_obs = next_share_obs[agent_id]
                if not self.use_centralized_Q1:  # default: True
                    agent_next_share_obs = next_obs[agent_id]
                for i in range(agent_next_obs.shape[0]):
                    step = counters[agent_id]
                    self.mappo_edge_buffer[agent_id].rewards[step] = rewards[agent_id][i]
                    self.mappo_edge_buffer[agent_id].next_obs[step] = np.hstack((agent_next_obs[i].copy(), uav_actions_env[agent_id]))
                    self.mappo_edge_buffer[agent_id].next_share_obs[step] = np.hstack((agent_next_share_obs[i].copy(), uav_actions_env[agent_id]))
                    counters[agent_id] = (step + 1) % self.mappo_edge_buffer[agent_id].buffer_length

    def run(self, start_epi, end_epi):
        print("Start time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        metrics = self.clear_metrics()
        for epd in range(start_epi, end_epi):  # episode步
            print('||                            Episode:{}                            ||'.format(epd))

            if self.use_linear_lr_decay:
                for agent_id in range(self.num_uav_agents):
                    self.mappo_uav_trainer[agent_id].policy.lr_decay(epd, self.episode)
                for agent_id in range(self.num_edge_agents):
                    self.mappo_edge_trainer[agent_id].policy.lr_decay(epd, self.episode)

            # record metrics
            epd_metrics = self.clear_epd_metrics()
            # 加载模拟数据
            start_time = self.sequence_keys[epd]
            one_sequence = self.simulation_sequences[start_time]

            all_task_num = 0  # 1个episode中任务总数
            edge_task_nums = [0] * self.num_edge_agents  # 1个episode中每个edge的任务总数--统计每个edge的流量
            edge_comp_task_nums = [0] * self.num_edge_agents  # 1个episode中卸载/迁移到每个edge的任务数
            uav_comp_task_nums = [0] * self.num_uav_agents  # 1个episode中迁移到每个uav的任务数
            uav_epi_rewards = [0] * self.num_uav_agents  # 1个episode中每个uav的总rewards
            uav_decision_counts = {i: [0] * self.num_uav_cover[i] for i in range(self.num_uav_agents)}  # 1个episode中每个uav决策覆盖到不同edge的次数
            counters_mappo_uav = [0] * self.num_uav_agents
            counters_mappo_edge = [0] * self.num_edge_agents

            # 每个episode的time_slots步
            for current_step in range(self.time_slots-1):
                if current_step == 0:
                    # edge_vehicle_map --- {edge_0:[v_1,v_2,...., ],edge_1:[],...,edge_num:[]}
                    edge_vehicle_map = one_sequence[current_step]
                    future_trajectory = self.get_prediction_traj(start_time + current_step)
                    task_num_now, task_nums, uav_obs, uav_share_obs, edge_obs, edge_share_obs = self.envs.reset(edge_vehicle_map, future_trajectory)  # obs: dict
                all_task_num += task_num_now
                edge_task_nums = [edge_task_nums[i] + task_nums[i] for i in range(self.num_edge_agents)]
                # sample uav position actions
                uav_actions, uav_actions_env, uav_decision_counts = self.collect_uav(uav_obs, uav_share_obs, uav_decision_counts)
                # sample edge computation offloading & resource management actions
                edge_actions = self.collect_edge(edge_obs, edge_share_obs, uav_actions_env)
                # observe reward and next obs
                next_edge_vehicle_map = one_sequence[current_step+1]
                next_future_trajectory = self.get_prediction_traj(start_time + current_step + 1)
                task_num_now, task_nums, comp_task_nums_e, comp_task_nums_u, \
                uav_obs, uav_share_obs, uav_rewards, \
                edge_obs, edge_share_obs, edge_next_obs, edge_next_share_obs, edge_rewards, info, done = self.envs.step(
                    edge_actions, edge_vehicle_map, next_edge_vehicle_map, next_future_trajectory)
                edge_comp_task_nums = [edge_comp_task_nums[i] + comp_task_nums_e[i] for i in range(self.num_edge_agents)]
                uav_comp_task_nums = [uav_comp_task_nums[i] + comp_task_nums_u[i] for i in range(self.num_uav_agents)]
                uav_epi_rewards = [uav_epi_rewards[i] + uav_rewards[i] for i in range(self.num_uav_agents)]
                edge_vehicle_map = next_edge_vehicle_map
                # insert data into uav mappo buffer
                self.insert_mappo_uav(counters_mappo_uav, uav_obs, uav_share_obs, uav_rewards)
                # insert data into edge mappo buffer
                self.insert_mappo_edge(counters_mappo_edge, edge_next_obs, edge_next_share_obs, edge_rewards, uav_actions_env)
                # update metrics
                epd_metrics = self.update_epd_metrics(epd_metrics, edge_rewards, info)

            # compute return and update network
            self.compute(self.mappo_uav_buffer, self.mappo_uav_trainer, self.num_uav_agents)
            self.train_mappo(self.mappo_uav_buffer, self.mappo_uav_trainer, self.num_uav_agents)
            self.compute(self.mappo_edge_buffer, self.mappo_edge_trainer, self.num_edge_agents)
            self.train_mappo(self.mappo_edge_buffer, self.mappo_edge_trainer, self.num_edge_agents)

            # save model
            if epd != 0 and (epd % self.save_interval == 0 or epd == self.episode - 1):
                self.save_mappo(self.mappo_uav_trainer, self.num_uav_agents, self.save_dir0)
                self.save_mappo(self.mappo_edge_trainer, self.num_edge_agents, self.save_dir1)

            if epd % self.log_interval == 0:
                self.log_epd_metrics("train_mappo_mappo_", epd_metrics, all_task_num, edge_task_nums,
                                     edge_comp_task_nums, uav_comp_task_nums, uav_epi_rewards, uav_decision_counts, epd)

            metrics = self.update_metrics(metrics, epd_metrics, all_task_num,
                                          edge_task_nums, edge_comp_task_nums, uav_comp_task_nums)
            print("Complete time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        self.save_metrics(metrics)
        print("End time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    @torch.no_grad()
    def get_uav_actions(self, uav_obs, uav_test_rnn_states, uav_decision_counts):
        test_masks = np.ones((1, 1), dtype=np.float32)
        uav_actions = {}
        uav_actions_env = {}
        for uav_agent_id in range(self.num_uav_agents):
            self.mappo_uav_trainer[uav_agent_id].prep_rollout()
            uav_agent_obs = uav_obs[uav_agent_id]
            uav_action, uav_test_rnn_states = self.mappo_uav_trainer[uav_agent_id].policy.act(
                uav_agent_obs, uav_test_rnn_states, test_masks, deterministic=True)
            uav_action = _t2n(uav_action)[0]
            uav_action_env = np.eye(self.envs.mappo_uav_action_space[uav_agent_id].n)[uav_action]  # rearrange action
            uav_test_rnn_states = _t2n(uav_test_rnn_states)
            for agent_id in self.uav_edge_map[uav_agent_id]:
                uav_actions[agent_id] = uav_action
                uav_actions_env[agent_id] = uav_action_env
            uav_decision_counts[uav_agent_id][uav_action] += 1
        return uav_actions, uav_actions_env, uav_test_rnn_states, uav_decision_counts

    @torch.no_grad()
    def get_edge_actions(self, edge_obs, uav_actions_env, edge_test_rnn_states):
        test_masks = np.ones((1, 1), dtype=np.float32)
        edge_actions = {}
        for agent_id in range(self.num_edge_agents):
            self.mappo_edge_trainer[agent_id].prep_rollout()
            if agent_id in edge_obs.keys():
                agent_off_obs = edge_obs[agent_id]
                available_actions = np.ones((self.num_edge_agents + 2), dtype=np.float32)

                # 判断该edge被哪个uav所覆盖 uav_id
                for uav_id, covered_edges in self.uav_edge_map.items():
                    if agent_id in covered_edges:
                        break
                if self.uav_edge_map[uav_id][np.argmax(uav_actions_env[agent_id])] != agent_id:
                    available_actions[-1] = 0

                agent_edge_actions = []
                for agent_off_ob in agent_off_obs:
                    final_available_actions = [available_actions, None]
                    edge_action, edge_test_rnn_states = self.mappo_edge_trainer[agent_id].policy.act(
                        np.hstack((agent_off_ob.copy(), uav_actions_env[agent_id])),
                        edge_test_rnn_states, test_masks, final_available_actions, deterministic=True)
                    edge_action = _t2n(edge_action)
                    edge_test_rnn_states = _t2n(edge_test_rnn_states)
                    agent_edge_actions.append(edge_action)
                edge_actions[agent_id] = np.array(agent_edge_actions)
        return edge_actions, edge_test_rnn_states

    @torch.no_grad()
    def test(self, start_epi, end_epi):
        print("Start time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        metrics = self.clear_metrics()
        uav_test_rnn_states = np.zeros((1, self.recurrent_N0, self.hidden_size0), dtype=np.float32)
        edge_test_rnn_states = np.zeros((1, self.recurrent_N1, self.hidden_size1), dtype=np.float32)
        for epd in range(start_epi, end_epi):  # episode步
            print('||                            Episode:{}                            ||'.format(epd))

            # record metrics
            epd_metrics = self.clear_epd_metrics()
            # 加载模拟数据
            start_time = self.sequence_keys[epd]
            one_sequence = self.simulation_sequences[start_time]

            all_task_num = 0  # 1个episode中任务总数
            # 1个episode中每个edge的任务总数--统计每个edge的流量
            edge_task_nums = [0] * self.num_edge_agents
            # 1个episode中卸载/迁移到每个edge的任务数
            edge_comp_task_nums = [0] * self.num_edge_agents
            # 1个episode中迁移到每个uav的任务数
            uav_comp_task_nums = [0] * self.num_uav_agents
            # 1个episode中每个uav的总rewards
            uav_epi_rewards = [0] * self.num_uav_agents
            # 1个episode中每个uav决策覆盖到不同edge的次数
            uav_decision_counts = {i: [0] * self.num_uav_cover[i] for i in range(self.num_uav_agents)}

            # 每个episode的time_slots步
            for current_step in range(self.time_slots - 1):
                if current_step == 0:
                    # edge_vehicle_map --- {edge_0:[v_1,v_2,...., ],edge_1:[],...,edge_num:[]}
                    edge_vehicle_map = one_sequence[current_step]
                    future_trajectory = self.get_prediction_traj(start_time + current_step)
                    task_num_now, task_nums, uav_obs, _, edge_obs, _ = self.envs.reset(edge_vehicle_map, future_trajectory)   # obs: dict
                all_task_num += task_num_now
                edge_task_nums = [edge_task_nums[i] + task_nums[i] for i in range(self.num_edge_agents)]
                # get uav position actions
                uav_actions, uav_actions_env, uav_test_rnn_states, uav_decision_counts = self.get_uav_actions(uav_obs, uav_test_rnn_states, uav_decision_counts)
                # get edge computation offloading & resource management actions
                edge_actions, edge_test_rnn_states = self.get_edge_actions(edge_obs, uav_actions_env, edge_test_rnn_states)
                # observe reward and next obs
                next_edge_vehicle_map = one_sequence[current_step + 1]
                next_future_trajectory = self.get_prediction_traj(start_time + current_step + 1)
                task_num_now, task_nums, comp_task_nums_e, comp_task_nums_u, \
                uav_obs, _, uav_rewards, edge_obs, _, _, _, edge_rewards, info, done = self.envs.step(
                    edge_actions, edge_vehicle_map, next_edge_vehicle_map, next_future_trajectory)
                edge_comp_task_nums = [edge_comp_task_nums[i] + comp_task_nums_e[i] for i in range(self.num_edge_agents)]
                uav_comp_task_nums = [uav_comp_task_nums[i] + comp_task_nums_u[i] for i in range(self.num_uav_agents)]
                uav_epi_rewards = [uav_epi_rewards[i] + uav_rewards[i] for i in range(self.num_uav_agents)]
                edge_vehicle_map = next_edge_vehicle_map
                # update metrics
                epd_metrics = self.update_epd_metrics(epd_metrics, edge_rewards, info)

            if epd % self.log_interval == 0:
                self.log_epd_metrics("test_mappo_mappo_", epd_metrics, all_task_num, edge_task_nums,
                                     edge_comp_task_nums, uav_comp_task_nums, uav_epi_rewards, uav_decision_counts, epd)
            metrics = self.update_metrics(metrics, epd_metrics, all_task_num,
                                          edge_task_nums, edge_comp_task_nums, uav_comp_task_nums)
            print("Complete time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        self.save_metrics(metrics)

        _, _, _, _, avr_per_task, avr_comp_per_task, avr_comm_per_task, \
        avr_energy_per_task, avr_comp_energy_per_task, avr_comm_energy_per_task, failure_rate, local_rate, \
        uav_util_per_time, edge_util_per_time, vehicle_util_per_time, avr_reward = metrics
        print("Test average latency per task: {}, Test average computation latency per task:{},"
              "Test average communication latency per task: {}".format(
            np.mean(avr_per_task), np.mean(avr_comp_per_task), np.mean(avr_comm_per_task)))
        print("Test average energy per task: {}, Test average computation energy per task:{},"
              "Test average communication energy per task: {}".format(
            np.mean(avr_energy_per_task), np.mean(avr_comp_energy_per_task), np.mean(avr_comm_energy_per_task)))
        print("Test average task failure rate: {}, Test average local task rate:{} ".format(
            np.mean(failure_rate), np.mean(local_rate)))
        print("Test average uav util per time: {}, Test average edge util per time: {}, "
              "Test average vehicle util per time:{} ".format(
            np.mean(uav_util_per_time), np.mean(edge_util_per_time), np.mean(vehicle_util_per_time)))
        print("Test average reward: {}".format(np.mean(avr_reward)))
        print("End time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
