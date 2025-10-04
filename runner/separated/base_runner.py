import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from myutils.util import save_pkl
from myutils.mappo_utils.separated_buffer_edge import SeparatedReplayBuffer as SeparatedReplayBufferEdge
from myutils.mappo_utils.separated_buffer_uav import SeparatedReplayBuffer as SeparatedReplayBufferUAV
from algorithms.MAPPO_Edge.mappo import MAPPO as MAPPOEdge
from algorithms.MAPPO_Edge.MAPPOPolicy import MAPPOPolicy as MAPPOPolicyEdge
from algorithms.MAPPO_UAV.mappo import MAPPO as MAPPOUAV
from algorithms.MAPPO_UAV.MAPPOPolicy import MAPPOPolicy as MAPPOPolicyUAV


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.device = config["device"]
        self.uav_edge_map = config["uav_edge_map"]
        self.num_uav_cover = config["num_uav_cover"]
        self.num_edge_agents = self.all_args.num_edge
        self.num_uav_agents = self.all_args.num_uav

        self.save_interval = self.all_args.save_interval
        self.log_interval = self.all_args.log_interval
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay

        # mappo-uav
        self.algorithm_name0 = self.all_args.algorithm_name0
        self.use_centralized_Q0 = self.all_args.use_centralized_Q0
        self.hidden_size0 = self.all_args.hidden_size0
        self.recurrent_N0 = self.all_args.recurrent_N0

        # mappo-edge
        self.algorithm_name1 = self.all_args.algorithm_name1
        self.use_centralized_Q1 = self.all_args.use_centralized_Q1
        self.hidden_size1 = self.all_args.hidden_size1
        self.recurrent_N1 = self.all_args.recurrent_N1

        # dir
        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / "logs")
        self.metrics_dir = str(self.run_dir / "metrics")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)
        if self.all_args.stage == "train":
            self.save_dir0 = str(self.run_dir / "models" / "MAPPO_uav")
            if not os.path.exists(self.save_dir0):
                os.makedirs(self.save_dir0)
            self.save_dir1 = str(self.run_dir / "models" / "MAPPO_edge")
            if not os.path.exists(self.save_dir1):
                os.makedirs(self.save_dir1)

        # policies
        self.mappo_uav_policy = []
        self.mappo_edge_policy = []
        # UAV Agents
        for agent_id in range(self.num_uav_agents):
            mappo_uav_share_obs_space = (
                self.envs.mappo_uav_share_obs_space[agent_id]
                if self.use_centralized_Q0
                else self.envs.mappo_uav_obs_space[agent_id]
            )
            # MAPPO policy network - uav
            po_uav = MAPPOPolicyUAV(
                self.all_args,
                self.envs.mappo_uav_obs_space[agent_id],
                mappo_uav_share_obs_space,
                self.envs.mappo_uav_action_space[agent_id],
                device=self.device,
            )
            self.mappo_uav_policy.append(po_uav)
        # Edge Agents
        for agent_id in range(self.num_edge_agents):
            mappo_edge_share_obs_space = (
                self.envs.mappo_edge_share_obs_space[agent_id]
                if self.use_centralized_Q1
                else self.envs.mappo_edge_obs_space[agent_id]
            )
            # MAPPO policy network - edge
            po_edge1 = MAPPOPolicyEdge(
                self.all_args,
                self.envs.mappo_edge_obs_space[agent_id],
                mappo_edge_share_obs_space,
                self.envs.mappo_edge_action_space[agent_id],
                device=self.device,
            )
            self.mappo_edge_policy.append(po_edge1)

        if self.all_args.stage == "test":
            if self.all_args.model_dir0 is not None:
                self.restore(self.mappo_uav_policy, self.num_uav_agents, str(self.all_args.model_dir0 / "models" / "MAPPO_uav"))
            if self.all_args.model_dir1 is not None:
                self.restore(self.mappo_edge_policy, self.num_edge_agents, str(self.all_args.model_dir1 / "models" / "MAPPO_edge"))

        self.mappo_uav_trainer = []
        self.mappo_uav_buffer = []
        for agent_id in range(self.num_uav_agents):
            # algorithm
            tr_uav = MAPPOUAV(self.all_args, self.mappo_uav_policy[agent_id], device=self.device)
            # buffer
            mappo_uav_share_obs_space = (
                self.envs.mappo_uav_share_obs_space[agent_id]
                if self.use_centralized_Q0
                else self.envs.mappo_uav_obs_space[agent_id]
            )
            bu_uav = SeparatedReplayBufferUAV(
                self.all_args,
                self.envs.mappo_uav_obs_space[agent_id],
                mappo_uav_share_obs_space,
                self.envs.mappo_uav_action_space[agent_id],
            )
            self.mappo_uav_trainer.append(tr_uav)
            self.mappo_uav_buffer.append(bu_uav)

        self.mappo_edge_trainer = []
        self.mappo_edge_buffer = []
        for agent_id in range(self.num_edge_agents):
            # algorithm
            tr_edge1 = MAPPOEdge(self.all_args, self.mappo_edge_policy[agent_id], device=self.device)
            # buffer
            mappo_edge_share_obs_space = (
                self.envs.mappo_edge_share_obs_space[agent_id]
                if self.use_centralized_Q1
                else self.envs.mappo_edge_obs_space[agent_id]
            )
            bu_edge1 = SeparatedReplayBufferEdge(
                self.all_args,
                self.envs.mappo_edge_obs_space[agent_id],
                mappo_edge_share_obs_space,
                self.envs.mappo_edge_action_space[agent_id],
            )
            self.mappo_edge_trainer.append(tr_edge1)
            self.mappo_edge_buffer.append(bu_edge1)

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self, mappobuffer, mappotrainer, num_agents):
        for agent_id in range(num_agents):
            step = mappobuffer[agent_id].step
            if step > 0:
                mappotrainer[agent_id].prep_rollout()
                next_value = mappotrainer[agent_id].policy.get_values(
                    mappobuffer[agent_id].next_share_obs[step-1],
                    mappobuffer[agent_id].rnn_states_critic[step],
                    mappobuffer[agent_id].masks[step],
                )
                next_value = _t2n(next_value)
                mappobuffer[agent_id].compute_returns(next_value, mappotrainer[agent_id].value_normalizer)

    def train_mappo(self, mappobuffer, mappotrainer, num_agents):
        train_infos = {}
        for agent_id in range(num_agents):
            if mappobuffer[agent_id].step > 0:
                mappotrainer[agent_id].prep_training()
                train_info = mappotrainer[agent_id].train(mappobuffer[agent_id])
                train_infos[agent_id] = train_info
                mappobuffer[agent_id].after_update()
        return train_infos

    def save_mappo(self, mappotrainer, num_agents, save_dir):
        for agent_id in range(num_agents):
            # mappo
            mappo_actor = mappotrainer[agent_id].policy.actor
            torch.save(mappo_actor.state_dict(), str(save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            mappo_critic = mappotrainer[agent_id].policy.critic
            torch.save(mappo_critic.state_dict(), str(save_dir) + "/critic_agent" + str(agent_id) + ".pt")

    def restore(self, policy, num_agents, model_dir):
        for agent_id in range(num_agents):
            policy_actor_state_dict = torch.load(str(model_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(model_dir) + "/critic_agent" + str(agent_id) + ".pt")
            policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_epd_metrics(self, tag, epd_metrics, all_task_num, edge_task_nums,
                        edge_comp_task_nums, uav_comp_task_nums, uav_epi_rewards, uav_decision_counts, epd):
        uav_utils, edge_utils, vehicle_utils, all_edge_rewards, \
        all_latency, all_wait_latency, all_exe_latency, all_tran_latency, all_mig_latency, \
        all_energy, all_store_energy, all_exe_energy, all_tran_energy, all_mig_energy, \
        all_failure_num, all_local_num = epd_metrics
        self.writer.add_scalars(tag+"total_task_num", {"total_task_num": all_task_num}, epd)
        for i in range(self.num_uav_agents):
            self.writer.add_scalars(tag + "uav_comp_task_num", {"uav%i_comp_task_num_per_slot" % i: uav_comp_task_nums[i]/(self.time_slots-1)}, epd)
            self.writer.add_scalars(tag + "uav_reward", {"uav%i_reward_per_slot" % i: uav_epi_rewards[i]/(self.time_slots-1)}, epd)
        for i in range(self.num_edge_agents):
            self.writer.add_scalars(tag + "edge_task_num", {"edge%i_task_num_per_slot" % i: edge_task_nums[i]/(self.time_slots-1)}, epd)
            self.writer.add_scalars(tag + "edge_comp_task_num", {"edge%i_comp_task_num_per_slot" % i: edge_comp_task_nums[i]/(self.time_slots-1)}, epd)
        for i in range(self.num_uav_agents):
            for j in range(self.num_uav_cover[i]):
                self.writer.add_scalars(tag + "uav%i_decision_counts" % i, {"edge%i_counts" % j: uav_decision_counts[i][j]/(self.time_slots-1)}, epd)
        self.writer.add_scalars(tag+"latency", {"average_completion_latency_per_task": all_latency / all_task_num}, epd)
        self.writer.add_scalars(tag+"latency", {"average_wait_latency_per_task": all_wait_latency / all_task_num}, epd)
        self.writer.add_scalars(tag+"latency", {"average_exe_latency_per_task": all_exe_latency / all_task_num}, epd)
        self.writer.add_scalars(tag+"latency", {"average_tran_latency_per_task": all_tran_latency / all_task_num}, epd)
        self.writer.add_scalars(tag+"latency", {"average_mig_latency_per_task": all_mig_latency / all_task_num}, epd)
        self.writer.add_scalars(tag+"energy", {"average_energy_per_task": all_energy / all_task_num}, epd)
        self.writer.add_scalars(tag+"energy", {"average_store_energy_per_task": all_store_energy / all_task_num}, epd)
        self.writer.add_scalars(tag+"energy", {"average_exe_energy_per_task": all_exe_energy / all_task_num}, epd)
        self.writer.add_scalars(tag+"energy", {"average_tran_energy_per_task": all_tran_energy / all_task_num}, epd)
        self.writer.add_scalars(tag+"energy", {"average_mig_energy_per_task": all_mig_energy / all_task_num}, epd)
        self.writer.add_scalars(tag+"rate", {"task_failure_rate": all_failure_num / all_task_num}, epd)
        self.writer.add_scalars(tag+"rate", {"local_task_rate": all_local_num / all_task_num}, epd)
        self.writer.add_scalars(tag+"ratio", {"average_utilization_of_uav_computation_resources": np.mean(uav_utils)}, epd)
        self.writer.add_scalars(tag+"ratio", {"average_utilization_of_edge_computation_resources": np.mean(edge_utils)}, epd)
        self.writer.add_scalars(tag+"ratio", {"average_utilization_of_vehicle_computation_resources": np.mean(vehicle_utils)}, epd)
        self.writer.add_scalars(tag+"reward", {"average_reward_per_task": all_edge_rewards / all_task_num}, epd)
        print('average completion latency per task: %.4f, average computation latency per task: %.4f,average communication latency per task: %.4f' %
              (all_latency / all_task_num, (all_wait_latency+all_exe_latency) / all_task_num, (all_tran_latency+all_mig_latency) / all_task_num))
        print('average energy per task: %.4f, average computation energy per task: %.4f, average communication energy per task: %.4f' %
              (all_energy / all_task_num, (all_store_energy+all_exe_energy) / all_task_num, (all_tran_energy+all_mig_energy) / all_task_num))
        print('task failure rate: %.4f, local task rate: %.4f' % (all_failure_num / all_task_num, all_local_num / all_task_num))
        print('average utilization of uav computation resources: %.4f' % (np.mean(uav_utils)))
        print('average utilization of edge computation resources: %.4f' % (np.mean(edge_utils)))
        print('average utilization of vehicle computation resources: %.4f' % (np.mean(vehicle_utils)))

    def clear_epd_metrics(self):
        uav_utils = []
        edge_utils = []  # 1个episode中time_slots步每一步的平均edge利用率
        vehicle_utils = []  # 1个episode中time_slots步每一步的平均vehicle利用率
        all_edge_rewards = 0  # 1个episode中time_slots步所有edges的总rewards
        all_latency = 0  # 1个episode中time_slots步所有edges的总时延
        all_wait_latency = 0  # 1个episode中time_slots步所有edges的等待时延
        all_exe_latency = 0
        all_tran_latency = 0
        all_mig_latency = 0
        all_energy = 0
        all_store_energy = 0
        all_exe_energy = 0
        all_tran_energy = 0
        all_mig_energy = 0
        all_failure_num = 0  # 1个episode中time_slots步所有edges的总失败任务数
        all_local_num = 0  # 1个episode中time_slots步所有edges的总本地执行任务数
        return uav_utils, edge_utils, vehicle_utils, all_edge_rewards, \
               all_latency, all_wait_latency, all_exe_latency, all_tran_latency, all_mig_latency, \
               all_energy, all_store_energy, all_exe_energy, all_tran_energy, all_mig_energy, all_failure_num, all_local_num

    def update_epd_metrics(self, epd_metrics, edge_rewards, info):
        uav_utils, edge_utils, vehicle_utils, all_edge_rewards, \
        all_latency, all_wait_latency, all_exe_latency, all_tran_latency, all_mig_latency,\
        all_energy, all_store_energy, all_exe_energy, all_tran_energy, all_mig_energy, \
        all_failure_num, all_local_num = epd_metrics
        uav_utils.append(info["average_uav_utilization"])
        edge_utils.append(info["average_edge_utilization"])
        vehicle_utils.append(info["average_vehicle_utilization"])
        all_edge_rewards += np.sum([np.sum(x) for x in edge_rewards.values()])  # 加上所有edges的总rewards
        all_latency += np.sum([x for x in info["latencies"].values()])  # 加上所有edges的时延
        all_wait_latency += np.sum([x for x in info["wait_latencies"].values()])  # 加上所有edges的等待时延
        all_exe_latency += np.sum([x for x in info["exe_latencies"].values()])
        all_tran_latency += np.sum([x for x in info["tran_latencies"].values()])
        all_mig_latency += np.sum([x for x in info["mig_latencies"].values()])
        all_energy += np.sum([x for x in info["energies"].values()])
        all_store_energy += np.sum([x for x in info["store_energies"].values()])
        all_exe_energy += np.sum([x for x in info["exe_energies"].values()])
        all_tran_energy += np.sum([x for x in info["tran_energies"].values()])
        all_mig_energy += np.sum([x for x in info["mig_energies"].values()])
        all_failure_num += np.sum([x for x in info["failure_nums"].values()])  # 加上所有edges的失败任务数
        all_local_num += np.sum([x for x in info["local_nums"].values()])
        return uav_utils, edge_utils, vehicle_utils, all_edge_rewards, \
               all_latency, all_wait_latency, all_exe_latency, all_tran_latency, all_mig_latency, \
               all_energy, all_store_energy, all_exe_energy, all_tran_energy, all_mig_energy, all_failure_num, all_local_num

    def clear_metrics(self):
        overall_task_num, overall_edge_task_nums, overall_edge_comp_task_nums, overall_uav_comp_task_nums, \
        avr_per_task, avr_comp_per_task, avr_comm_per_task, \
        avr_energy_per_task, avr_comp_energy_per_task, avr_comm_energy_per_task, failure_rate, local_rate, \
        uav_util_per_time, edge_util_per_time, vehicle_util_per_time, avr_reward = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        return overall_task_num, overall_edge_task_nums, overall_edge_comp_task_nums, overall_uav_comp_task_nums, \
               avr_per_task, avr_comp_per_task, avr_comm_per_task, \
               avr_energy_per_task, avr_comp_energy_per_task, avr_comm_energy_per_task, failure_rate, local_rate, \
               uav_util_per_time, edge_util_per_time, vehicle_util_per_time, avr_reward

    def update_metrics(self, metrics, epd_metrics, all_task_num, edge_task_nums, edge_comp_task_nums, uav_comp_task_nums):
        overall_task_num, overall_edge_task_nums, overall_edge_comp_task_nums, overall_uav_comp_task_nums, \
        avr_per_task, avr_comp_per_task, avr_comm_per_task, \
        avr_energy_per_task, avr_comp_energy_per_task, avr_comm_energy_per_task, failure_rate, local_rate, \
        uav_util_per_time, edge_util_per_time, vehicle_util_per_time, avr_reward = metrics
        uav_utils, edge_utils, vehicle_utils, all_edge_rewards, \
        all_latency, all_wait_latency, all_exe_latency, all_tran_latency, all_mig_latency, \
        all_energy, all_store_energy, all_exe_energy, all_tran_energy, all_mig_energy, all_failure_num, all_local_num = epd_metrics

        overall_task_num.append(all_task_num)
        overall_edge_task_nums.append(edge_task_nums)
        overall_edge_comp_task_nums.append(edge_comp_task_nums)
        overall_uav_comp_task_nums.append(uav_comp_task_nums)
        avr_per_task.append(all_latency / all_task_num)
        avr_comp_per_task.append((all_wait_latency + all_exe_latency) / all_task_num)
        avr_comm_per_task.append((all_tran_latency + all_mig_latency) / all_task_num)
        avr_energy_per_task.append(all_energy / all_task_num)
        avr_comp_energy_per_task.append((all_store_energy + all_exe_energy) / all_task_num)
        avr_comm_energy_per_task.append((all_tran_energy + all_mig_energy) / all_task_num)
        failure_rate.append(all_failure_num / all_task_num)
        local_rate.append(all_local_num / all_task_num)
        uav_util_per_time.append(np.mean(uav_utils))
        edge_util_per_time.append(np.mean(edge_utils))
        vehicle_util_per_time.append(np.mean(vehicle_utils))
        avr_reward.append(all_edge_rewards / all_task_num)
        return overall_task_num, overall_edge_task_nums, overall_edge_comp_task_nums, overall_uav_comp_task_nums, \
               avr_per_task, avr_comp_per_task, avr_comm_per_task, \
               avr_energy_per_task, avr_comp_energy_per_task, avr_comm_energy_per_task, failure_rate, local_rate, \
               uav_util_per_time, edge_util_per_time, vehicle_util_per_time, avr_reward

    def save_metrics(self, metrics):
        overall_task_num, overall_edge_task_nums, overall_edge_comp_task_nums, overall_uav_comp_task_nums, \
        avr_per_task, avr_comp_per_task, avr_comm_per_task, \
        avr_energy_per_task, avr_comp_energy_per_task, avr_comm_energy_per_task, failure_rate, local_rate, \
        uav_util_per_time, edge_util_per_time, vehicle_util_per_time, avr_reward = metrics

        save_pkl(self.metrics_dir + "/overall_task_num.pkl", overall_task_num)
        save_pkl(self.metrics_dir + "/overall_edge_task_nums.pkl", overall_edge_task_nums)
        save_pkl(self.metrics_dir + "/overall_edge_comp_task_nums.pkl", overall_edge_comp_task_nums)
        save_pkl(self.metrics_dir + "/overall_uav_comp_task_nums.pkl", overall_uav_comp_task_nums)
        save_pkl(self.metrics_dir + "/latency.pkl", avr_per_task)
        save_pkl(self.metrics_dir + "/computation_latency.pkl", avr_comp_per_task)
        save_pkl(self.metrics_dir + "/communication_latency.pkl", avr_comm_per_task)
        save_pkl(self.metrics_dir + "/energy.pkl", avr_energy_per_task)
        save_pkl(self.metrics_dir + "/computation_energy.pkl", avr_comp_energy_per_task)
        save_pkl(self.metrics_dir + "/communication_energy.pkl", avr_comm_energy_per_task)
        save_pkl(self.metrics_dir + "/failure_rate.pkl", failure_rate)
        save_pkl(self.metrics_dir + "/local_rate.pkl", local_rate)
        save_pkl(self.metrics_dir + "/uav_util.pkl", uav_util_per_time)
        save_pkl(self.metrics_dir + "/edge_util.pkl", edge_util_per_time)
        save_pkl(self.metrics_dir + "/vehicle_util.pkl", vehicle_util_per_time)
        save_pkl(self.metrics_dir + "/reward.pkl", avr_reward)
