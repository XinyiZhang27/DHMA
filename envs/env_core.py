import numpy as np
import random

class UAV:
    def __init__(self, uav_id, num_uav, covered_edges, cpu, core, bw):
        self.uav_id = uav_id
        self.num_uav = num_uav
        self.covered_edges = covered_edges

        self.default_settings = {"cpu": cpu, "core": core, "bw": bw, "buff_CPU": 0}
        # edge 状态(4)
        self.cpu = cpu  # 算力
        self.core = core
        self.bw = bw
        self.buff_CPU = 0  # 任务量队列(以所需CPU时间为单位)

        # execution energy coefficient
        self.kappa = 0.1
        # storage energy coefficient
        self.varepsilon = 0.01
        # vehicle-uav信道增益
        self.channel_gain = 50
        # vehicle-uav信道噪声建模
        self.noise = 2 / np.power(10, 13)

    def reset(self):
        self.cpu = self.default_settings["cpu"]
        self.core = self.default_settings["core"]
        self.bw = self.default_settings["bw"]
        self.buff_CPU = self.default_settings["buff_CPU"]  # 任务量队列(以所需CPU时间为单位)


class Edge:
    def __init__(self, edge_id, num_edge, trans_rate_edge, cpu, core, bw):

        self.edge_id = edge_id
        self.num_edge = num_edge

        self.default_settings = {"cpu": cpu, "core": core, "bw": bw, "buff_CPU": 0}
        # edge 状态(3)
        self.cpu = cpu   # edge server 算力
        self.core = core
        self.bw = bw     # edge server 带宽
        self.buff_CPU = 0  # 任务量队列(以所需CPU时间为单位)

        # execution energy coefficient
        self.kappa = 0.1
        # storage energy coefficient
        self.varepsilon = 0.01
        self.trans = 2.5  # 传输功率
        # vehicle-edge信道增益
        self.channel_gain = 50
        # vehicle-edge信道噪声建模
        self.noise = 2 / np.power(10, 13)
        # edge之间mig的数据传输速率
        self.trans_rate_edge = trans_rate_edge

        # 覆盖车辆
        self.connected_vehicles = []
        # 预测值
        self.predict_trajectory = []

    def reset(self):
        self.cpu = self.default_settings["cpu"]  # edge server 算力
        self.core = self.default_settings["core"]
        self.bw = self.default_settings["bw"]  # edge server 带宽
        self.buff_CPU = self.default_settings["buff_CPU"]  # 任务量队列(以所需CPU时间为单位)
        self.connected_vehicles = []  # 清空连接车辆
        self.predict_trajectory = []  # 重置预测值

    def vehicle_connection(self, vehicles):
        self.connected_vehicles = vehicles

    def clear_futuretrajectory(self):
        self.predict_trajectory = []

    def update_futuretrajectory(self, trajectory):
        self.predict_trajectory.append(trajectory)


class Task:
    def __init__(self, data_size, CPU_cycle, ddl):
        # 任务状态(3)
        self.data_size = data_size
        self.CPU_cycle = CPU_cycle
        self.ddl = ddl


class Priority_Task:
    def __init__(self, vehicle, priority):
        self.vehicle = vehicle
        self.priority = priority


class Task_pool:
    def __init__(self, task_parameter):
        self.task_param = task_parameter  # [data_size, CPU_cycle, DDl]

    def random_sample_n(self, size):
        task_list = []
        for _ in range(size):
            datasize = random.uniform(self.task_param[0][0], self.task_param[0][1])
            cpucycles = random.uniform(self.task_param[1][0], self.task_param[1][1])
            ddl = random.uniform(self.task_param[2][0], self.task_param[2][1])
            one_task = Task(datasize, cpucycles, ddl)
            task_list.append(one_task)
        return task_list


class Vehicle:
    def __init__(self, vehicle_id, cpu=2, core=1):

        self.vehicle_id = vehicle_id
        # vehicle状态(2)
        self.cpu = cpu   # 算力
        self.core = core  # 核数
        self.buff_CPU = 0  # 任务量队列(以所需CPU时间为单位)

        self.kappa = 0.1  # execution energy coefficient
        self.varepsilon = 0.01  # storage energy coefficient
        self.trans = 2  # 传输功率

    def generate_task(self, task_pool):
        one_task = task_pool.random_sample_n(size=1)[0]
        self.current_task = one_task


class EnvCore(object):
    def __init__(self, all_args, Edge_IDs, edge_index_map, uav_edge_map, shortest_paths, num_uav_cover, time_slot, pred_len=8):
        self.num_edge = all_args.num_edge
        self.num_uav = all_args.num_uav

        # 设置UAV
        self.uavs = {}
        for i in range(0, self.num_uav):
            uav_i = UAV(i, self.num_uav, uav_edge_map[i],
                        cpu=all_args.cpu_uav, core=all_args.core_uav, bw=all_args.bw_uav)
            self.uavs[i] = uav_i

        # 设置edge
        self.edges = {}
        for i in Edge_IDs:
            edge_i = Edge(i, self.num_edge, all_args.trans_rate_edge,
                          cpu=all_args.cpu_rsu, core=all_args.core_rsu, bw=all_args.bw_rsu)
            self.edges[i] = edge_i

        self.coef1 = all_args.coef1
        self.coef2 = all_args.coef2

        self.Edge_IDs = Edge_IDs
        self.Edge_index_map = edge_index_map
        self.shortest_paths = shortest_paths
        self.num_uav_cover = num_uav_cover
        self.time_slots = time_slot  # 强化学习评估时长（一分钟）
        self.pred_len = pred_len  # 预测轨迹长度

        # 任务建模
        self.task_pool = Task_pool(all_args.task_parameter)
        # 车辆建模
        self.vehicle_pool = {}  # 系统内当前车辆，vehicle_ID 为 key

        # UAV 位置规划 MAPPO
        self.obs_uav_dim = [x*(8+self.pred_len) + 4 for x in self.num_uav_cover]
        self.action_uav_dim = self.num_uav_cover
        self.share_obs_uav_dim = self.num_edge*(8+self.pred_len) + 4

        # Edge 卸载+分配 MAPPO
        self.action_discrete_dim = self.num_edge + 1 + 1  # (offR + migR) + offA + local
        self.action_continuous_dim = 2  # bandwidth ratio + priority score
        self.obs_edge_dim = []
        self.share_obs_edge_dim = []
        for id_i, edge_i in self.edges.items():
            agent_id = self.Edge_index_map[id_i]
            for uav_id, uav_i in self.uavs.items():
                if agent_id in uav_i.covered_edges:
                    break
            self.obs_edge_dim.append(7 + self.pred_len + 5 + 4 + self.num_uav_cover[uav_id])
            self.share_obs_edge_dim.append(7 + self.pred_len + self.num_edge*5 + 4 + self.num_uav_cover[uav_id])

    def reset(self, edge_vehicle_map, future_trajectory):
        all_task_num = 0
        # 未来每个edge覆盖的车辆数--预测每个edge的流量
        future_edge_task_nums = np.zeros([self.num_edge, self.pred_len])
        # 每个edge覆盖的车辆数--统计每个edge的流量
        edge_task_nums = [0] * self.num_edge
        # 每个edge覆盖的任务数据总量
        edge_task_data_size = [0] * self.num_edge
        # 每个edge覆盖的任务cpu_cycle总数
        edge_task_CPU_cycle = [0] * self.num_edge

        # 重置系统内当前车辆,尚未考虑多个连续episode之间vehicle_pool的保留
        self.vehicle_pool = {}
        # 重置每个UAV的建模
        for _, uav_i in self.uavs.items():
            uav_i.reset()
        # 重置每个edge的建模
        for _, edge_i in self.edges.items():
            edge_i.reset()

        uav_obs = {}
        uav_share_obs = {}
        edge_obs = {}
        edge_share_obs = {}

        whole_edge_state = []
        for _, edge_i in self.edges.items():
            whole_edge_state.append([edge_i.cpu, edge_i.core, edge_i.buff_CPU, edge_i.bw, edge_i.trans])
        whole_edge_state = np.array(whole_edge_state).flatten()

        # 对于每一个edge
        for id_i, edge_i in self.edges.items():
            agent_id = self.Edge_index_map[id_i]
            # 车辆根据模拟数据加入vehicle_pool并与Edges连接，更新连接车辆的轨迹预测值
            # edge_vehicle_map --- {edge_0:[v_1,v_2,...., ],edge_1:[],...,edge_num:[]}
            if id_i in edge_vehicle_map:
                vehicle_ids = edge_vehicle_map[id_i]
                for v_id in vehicle_ids:
                    vehicle_i = Vehicle(v_id)  # 新建一个车辆
                    self.vehicle_pool[v_id] = vehicle_i  # 加入系统的车辆pool中
                    if future_trajectory[v_id] != None:
                        edge_i.update_futuretrajectory(
                            np.array([self.Edge_index_map[str(int(item)) if isinstance(item, float) else item]
                                      for item in future_trajectory[v_id]]))  # 更新预测值
                    else:
                        edge_i.update_futuretrajectory(np.full(self.pred_len, -1))
                all_task_num += len(vehicle_ids)
                edge_task_nums[agent_id] = len(vehicle_ids)  # 统计每个edge的流量
                edge_i.vehicle_connection(vehicle_ids)  # ID 记录至对应edge

            # 每个车辆生成任务
            for v_id in edge_i.connected_vehicles:
                v_i = self.vehicle_pool[v_id]
                v_i.generate_task(self.task_pool)

            # Edge Information
            edge_state = np.array([edge_i.cpu, edge_i.core, edge_i.buff_CPU, edge_i.bw, edge_i.trans])
            # UAV Information
            for _, uav_i in self.uavs.items():
                if agent_id in uav_i.covered_edges:
                    uav_state = np.array([uav_i.cpu, uav_i.core, uav_i.buff_CPU, uav_i.bw])
                    break

            if len(edge_i.connected_vehicles) > 0:
                edge_obs[agent_id] = []
                edge_share_obs[agent_id] = []
                # Vehicle information + Task information
                for i, v_id in enumerate(edge_i.connected_vehicles):
                    v_i = self.vehicle_pool[v_id]
                    edge_task_data_size[agent_id] += v_i.current_task.data_size
                    edge_task_CPU_cycle[agent_id] += v_i.current_task.CPU_cycle
                    # [车辆算力，车辆任务量队列(CPU)，传输功率；任务数据量，任务所需CPU，任务截止时间]
                    vehicle_task_state = [v_i.cpu, v_i.core, v_i.buff_CPU, v_i.trans,
                                          v_i.current_task.data_size, v_i.current_task.CPU_cycle, v_i.current_task.ddl]
                    prediction_state = edge_i.predict_trajectory[i]
                    edge_obs[agent_id].append(np.hstack((vehicle_task_state, prediction_state, edge_state, uav_state)))
                    edge_share_obs[agent_id].append(np.hstack((vehicle_task_state, prediction_state, whole_edge_state, uav_state)))
                edge_obs[agent_id] = np.array(edge_obs[agent_id])
                edge_share_obs[agent_id] = np.array(edge_share_obs[agent_id])

            # 统计每个edge的未来流量
            if edge_i.predict_trajectory != []:
                edge_predict_trajectories = np.array(edge_i.predict_trajectory)
                future_edge_task_nums += np.apply_along_axis(lambda x: np.bincount(x, minlength=self.num_edge+1), axis=0,
                     arr=np.where(edge_predict_trajectories == -1, self.num_edge, edge_predict_trajectories))[:-1, :]

        whole_edge_state = []
        whole_edge_prediction_state = []
        for id_i, edge_i in self.edges.items():
            agent_id = self.Edge_index_map[id_i]
            whole_edge_state.append([edge_i.cpu, edge_i.core, edge_i.buff_CPU, edge_i.bw, edge_i.trans,
                                     edge_task_nums[agent_id], edge_task_data_size[agent_id], edge_task_CPU_cycle[agent_id]])
            whole_edge_prediction_state.append(future_edge_task_nums[agent_id])
        whole_edge_state = np.array(whole_edge_state).flatten()
        whole_edge_prediction_state = np.array(whole_edge_prediction_state).flatten()
        whole_edge_state = np.hstack((np.array(whole_edge_state), np.array(whole_edge_prediction_state))).flatten()

        # 对于每一个UAV
        for uav_id, uav_i in self.uavs.items():
            # UAV Information
            uav_state = np.array([uav_i.cpu, uav_i.core, uav_i.buff_CPU, uav_i.bw])
            edge_state = []
            edge_prediction_state = []
            for agent_id in uav_i.covered_edges:
                for id_i, edge_i in self.edges.items():
                    if self.Edge_index_map[id_i] == agent_id:
                        break
                edge_state.append([edge_i.cpu, edge_i.core, edge_i.buff_CPU, edge_i.bw, edge_i.trans, edge_task_nums[agent_id],
                                   edge_task_data_size[agent_id], edge_task_CPU_cycle[agent_id]])
                edge_prediction_state.append(future_edge_task_nums[agent_id])
            edge_state = np.hstack((np.array(edge_state), np.array(edge_prediction_state))).flatten()
            uav_obs[uav_id] = np.hstack((edge_state, uav_state))
            uav_share_obs[uav_id] = np.hstack((whole_edge_state, uav_state))

        self.done = False  # 模拟终止flag
        self.count_step = 0

        return all_task_num, edge_task_nums, uav_obs, uav_share_obs, edge_obs, edge_share_obs

    def step(self, edge_actions, edge_vehicle_map, next_edge_vehicle_map, next_future_trajectory):
        latencies = {}  # 所有edges的总时延
        wait_latencies = {}  # 所有edges的等待时延
        tran_latencies = {}
        exe_latencies = {}
        mig_latencies = {}
        energies = {}
        store_energies = {}
        tran_energies = {}
        exe_energies = {}
        mig_energies = {}
        edge_rewards = {}  # 所有edges的rewards
        uav_rewards = {}  # 所有uavs的rewards
        failure_nums = {}  # 所有edges的失败数
        local_nums = {}  # 所有edges的本地执行任务数

        vehicle_resource_utilization = []  # 每一辆车
        edge_resource_utilization = []  # 每个edge
        uav_resource_utilization = []  # 每个uav

        # 卸载/迁移到每个edge的任务数--用于显示负载均衡
        edge_comp_task_nums = [0] * self.num_edge
        # 卸载到每个uav的任务数
        uav_comp_task_nums = [0] * self.num_uav

        # 对决策进行预处理
        # 1) 统计卸载到相同的uav/edge的任务的bw，用于后续计算bw
        # 2) 统计卸载到相同的uav/edge的任务，用于后续根据priority给同一队列任务排序，计算等待时延
        rearrange_edge_bws = {}  # 卸载到RSU/迁移到RSU
        rearrange_uav_bws = {}  # 卸载到UAV
        tasks_edge = {}  # 卸载到RSU/迁移到RSU
        tasks_uav = {}  # 卸载到UAV
        for id_i, edge_i in self.edges.items():
            agent_id = self.Edge_index_map[id_i]
            # 判断该edge被哪个uav所覆盖 uav_id
            for uav_id, uav_i in self.uavs.items():
                if agent_id in uav_i.covered_edges:
                    break
            if agent_id in edge_actions.keys():
                for i, v_id in enumerate(edge_i.connected_vehicles):
                    vehicle_i = self.vehicle_pool[v_id]
                    if edge_actions[agent_id][i][0] == self.num_edge:  # 本地
                        continue
                    if edge_actions[agent_id][i][0] == self.num_edge + 1:  # 卸载到UAV
                        if uav_id not in rearrange_uav_bws.keys():
                            rearrange_uav_bws[uav_id] = []
                            tasks_uav[uav_id] = []
                        rearrange_uav_bws[uav_id].append(edge_actions[agent_id][i][1])  # bandwidth
                        tasks_uav[uav_id].append(Priority_Task(vehicle_i, edge_actions[agent_id][i][2]))  # priority
                    else:  # 卸载到RSU/迁移到RSU
                        if agent_id not in rearrange_edge_bws.keys():
                            rearrange_edge_bws[agent_id] = []
                        rearrange_edge_bws[agent_id].append(edge_actions[agent_id][i][1])  # bandwidth
                        if edge_actions[agent_id][i][0] not in tasks_edge.keys():
                            tasks_edge[edge_actions[agent_id][i][0]] = []
                        tasks_edge[edge_actions[agent_id][i][0]].append(Priority_Task(vehicle_i, edge_actions[agent_id][i][2]))  # priority

        # 按照priority排序
        for uav_id in tasks_uav.keys():
            tasks_uav[uav_id] = sorted(tasks_uav[uav_id], key=lambda priority_task: priority_task.priority, reverse=True)  # 降序
        for agent_id in tasks_edge.keys():
            tasks_edge[agent_id] = sorted(tasks_edge[agent_id], key=lambda priority_task: priority_task.priority, reverse=True)  # 降序

        # 计算排序后每个任务之前有多少 CPU cycles
        cpu_cycles_before_task = {}
        for uav_id, priority_tasks in tasks_uav.items():
            uav_i = self.uavs[uav_id]
            for priority_task in priority_tasks:
                cpu_cycles_before_task[priority_task.vehicle] = uav_i.buff_CPU
                uav_i.buff_CPU += priority_task.vehicle.current_task.CPU_cycle
        for agent_id, priority_tasks in tasks_edge.items():
            id_i = next((id_i for id_i, edge_i in self.edges.items() if self.Edge_index_map[id_i] == agent_id), None)
            edge_i = self.edges[id_i]
            for priority_task in priority_tasks:
                cpu_cycles_before_task[priority_task.vehicle] = edge_i.buff_CPU
                edge_i.buff_CPU += priority_task.vehicle.current_task.CPU_cycle

        # 按照决策与环境交互，计算reward
        for id_i, edge_i in self.edges.items():
            agent_id = self.Edge_index_map[id_i]
            # 判断该edge被哪个uav所覆盖 uav_id
            for uav_id, uav_i in self.uavs.items():
                if agent_id in uav_i.covered_edges:
                    break
            if agent_id in edge_actions.keys():
                all_wait_latency = 0  # edge_i所有车辆的等待时延
                all_exe_latency = 0  # edge_i所有车辆的计算时延
                all_tran_latency = 0  # edge_i所有车辆的传输时延
                all_mig_latency = 0  # edge_i所有车辆的迁移时延
                all_latency = 0  # edge_i所有车辆的总时延
                all_store_energy = 0
                all_exe_energy = 0
                all_tran_energy = 0
                all_mig_energy = 0
                all_energy = 0
                failure_num = 0  # edge_i所有车辆任务中失败的个数
                local_num = 0  # edge_i所有车辆任务中本地执行的个数
                edge_agent_rewards = []

                edge_action = edge_actions[agent_id]

                for i, v_id in enumerate(edge_i.connected_vehicles):  # 对于每一辆车执行决策
                    vehicle_i = self.vehicle_pool[v_id]
                    # 不同卸载策略:
                    # 本地执行
                    if edge_action[i][0] == self.num_edge:
                        tran_latency, tran_energy = 0, 0
                        mig_latency, mig_energy = 0, 0
                        wait_latency, store_energy = 0, 0
                        if vehicle_i.buff_CPU > 0:  # 本地计算是否等待
                            wait_latency = vehicle_i.buff_CPU / vehicle_i.cpu
                            store_energy = vehicle_i.varepsilon * vehicle_i.current_task.data_size * wait_latency
                        exe_latency = vehicle_i.current_task.CPU_cycle / vehicle_i.cpu
                        exe_energy = vehicle_i.kappa * np.square(vehicle_i.cpu/vehicle_i.core) * vehicle_i.current_task.CPU_cycle
                        latency = tran_latency + mig_latency + wait_latency + exe_latency  # 总时延
                        energy = tran_energy + mig_energy + store_energy + exe_energy  # 总能耗
                        local_num += 1
                        # 更新车辆buff
                        vehicle_i.buff_CPU += vehicle_i.current_task.CPU_cycle
                    # 卸载到无人机
                    elif edge_action[i][0] == self.num_edge + 1:
                        rearrange_uav_bw = np.array(rearrange_uav_bws[uav_id])
                        ratio_bw = np.exp(edge_action[i][1]) / np.sum(np.exp(rearrange_uav_bw[:]))
                        data_rate = (uav_i.bw * ratio_bw) * np.log2(1 + vehicle_i.trans * uav_i.channel_gain / uav_i.noise)
                        tran_latency = vehicle_i.current_task.data_size / data_rate
                        tran_energy = vehicle_i.trans * tran_latency
                        mig_latency, mig_energy = 0, 0
                        wait_latency, store_energy = 0, 0
                        if cpu_cycles_before_task[vehicle_i] > 0:  # uav计算是否等待
                            wait_latency = cpu_cycles_before_task[vehicle_i] / uav_i.cpu
                            store_energy = uav_i.varepsilon * vehicle_i.current_task.data_size * wait_latency
                        exe_latency = vehicle_i.current_task.CPU_cycle / uav_i.cpu
                        exe_energy = uav_i.kappa * np.square(uav_i.cpu/uav_i.core) * vehicle_i.current_task.CPU_cycle
                        latency = tran_latency + mig_latency + wait_latency + exe_latency
                        energy = tran_energy + mig_energy + store_energy + exe_energy  # 总能耗
                        uav_comp_task_nums[uav_id] += 1
                    # 卸载到RSU
                    elif edge_action[i][0] == agent_id:
                        rearrange_edge_bw = np.array(rearrange_edge_bws[agent_id])
                        ratio_bw = np.exp(edge_action[i][1]) / np.sum(np.exp(rearrange_edge_bw[:]))
                        data_rate = (edge_i.bw * ratio_bw) * np.log2(1 + vehicle_i.trans * edge_i.channel_gain / edge_i.noise)
                        tran_latency = vehicle_i.current_task.data_size / data_rate  # 传输时延与任务量有关
                        tran_energy = vehicle_i.trans * tran_latency
                        mig_latency, mig_energy = 0, 0
                        wait_latency, store_energy = 0, 0
                        if cpu_cycles_before_task[vehicle_i] > 0:  # 边缘计算是否等待
                            wait_latency = cpu_cycles_before_task[vehicle_i] / edge_i.cpu
                            store_energy = edge_i.varepsilon * vehicle_i.current_task.data_size * wait_latency
                        exe_latency = vehicle_i.current_task.CPU_cycle / edge_i.cpu  # 执行时延与edge算力有关
                        exe_energy = edge_i.kappa * np.square(edge_i.cpu/edge_i.core) * vehicle_i.current_task.CPU_cycle
                        latency = tran_latency + mig_latency + wait_latency + exe_latency
                        energy = tran_energy + mig_energy + store_energy + exe_energy  # 总能耗
                        edge_comp_task_nums[agent_id] += 1
                    # 迁移到其他edge
                    else:
                        next_edge_id = [key for key, value in self.Edge_index_map.items() if value == edge_action[i][0]][0]
                        next_edge_i = self.edges[next_edge_id]
                        next_agent_id = self.Edge_index_map[next_edge_id]
                        rearrange_edge_bw = np.array(rearrange_edge_bws[agent_id])
                        ratio_bw = np.exp(edge_action[i][1]) / np.sum(np.exp(rearrange_edge_bw[:]))
                        data_rate = (edge_i.bw * ratio_bw) * np.log2(1 + vehicle_i.trans * edge_i.channel_gain / edge_i.noise)
                        tran_latency = vehicle_i.current_task.data_size / data_rate
                        tran_energy = vehicle_i.trans * tran_latency
                        mig_latency = vehicle_i.current_task.data_size * self.shortest_paths[agent_id][
                            next_agent_id] / next_edge_i.trans_rate_edge
                        mig_energy = edge_i.trans * mig_latency
                        wait_latency, store_energy = 0, 0
                        if cpu_cycles_before_task[vehicle_i] > 0:  # 下一个edge边缘计算是否等待
                            wait_latency = cpu_cycles_before_task[vehicle_i] / next_edge_i.cpu
                            store_energy = next_edge_i.varepsilon * vehicle_i.current_task.data_size * wait_latency
                        exe_latency = vehicle_i.current_task.CPU_cycle / next_edge_i.cpu  # 在下一个edge的执行时延
                        exe_energy = next_edge_i.kappa * np.square(next_edge_i.cpu/next_edge_i.core) * vehicle_i.current_task.CPU_cycle
                        latency = tran_latency + mig_latency + wait_latency + exe_latency
                        energy = tran_energy + mig_energy + store_energy + exe_energy  # 总能耗
                        edge_comp_task_nums[next_agent_id] += 1

                    # 执行车辆的计算: 无论当前任务是否卸载到车辆执行，每个时刻车辆都要执行计算buff上的任务
                    if vehicle_i.buff_CPU < vehicle_i.cpu:
                        vehicle_resource_utilization.append(vehicle_i.buff_CPU / vehicle_i.cpu)
                        vehicle_i.buff_CPU = 0
                    else:
                        vehicle_resource_utilization.append(1)  # 如果执行之后, vehicle的buff不为0, vehicle资源利用率为1
                        vehicle_i.buff_CPU -= vehicle_i.cpu

                    all_latency += latency
                    all_wait_latency += wait_latency
                    all_exe_latency += exe_latency
                    all_tran_latency += tran_latency
                    all_mig_latency += mig_latency
                    all_energy += energy
                    all_store_energy += store_energy
                    all_exe_energy += exe_energy
                    all_tran_energy += tran_energy
                    all_mig_energy += mig_energy

                    # 判断该车任务是否惩罚, 并计算失败率
                    penalty = 0
                    if latency > vehicle_i.current_task.ddl:
                        penalty = latency - vehicle_i.current_task.ddl
                        failure_num += 1
                    edge_agent_rewards.append(-latency*self.coef1 - energy*self.coef2 - penalty)

                latencies[agent_id] = all_latency
                wait_latencies[agent_id] = all_wait_latency
                exe_latencies[agent_id] = all_exe_latency
                tran_latencies[agent_id] = all_tran_latency
                mig_latencies[agent_id] = all_mig_latency
                energies[agent_id] = all_energy
                store_energies[agent_id] = all_store_energy
                exe_energies[agent_id] = all_exe_energy
                tran_energies[agent_id] = all_tran_energy
                mig_energies[agent_id] = all_mig_energy
                edge_rewards[agent_id] = edge_agent_rewards
                failure_nums[agent_id] = failure_num
                local_nums[agent_id] = local_num

        # 计算每个UAV Agent的reward - 所覆盖edges的所有任务平均reward
        for uav_id, uav_i in self.uavs.items():
            uav_agent_rewards = []
            for agent_id in uav_i.covered_edges:
                if agent_id in edge_actions.keys():
                    uav_agent_rewards += edge_rewards[agent_id]
            if uav_agent_rewards != []:
                uav_rewards[uav_id] = sum(uav_agent_rewards)/len(uav_agent_rewards)
            else:
                uav_rewards[uav_id] = 0

        # 执行每一个edge的计算
        for id_i in self.Edge_IDs:
            edge_i = self.edges[id_i]
            if edge_i.buff_CPU < edge_i.cpu:
                edge_resource_utilization.append(edge_i.buff_CPU / edge_i.cpu)
                edge_i.buff_CPU = 0
            else:
                edge_resource_utilization.append(1)
                edge_i.buff_CPU -= edge_i.cpu
        # 执行每一个uav的计算
        for _, uav_i in self.uavs.items():
            if uav_i.buff_CPU < uav_i.cpu:
                uav_resource_utilization.append(uav_i.buff_CPU / uav_i.cpu)
                uav_i.buff_CPU = 0
            else:
                uav_resource_utilization.append(1)
                uav_i.buff_CPU -= uav_i.cpu

        # action执行完毕，进入下一时刻, 结合模拟数据调整任务
        all_task_num = 0
        # 未来每个edge覆盖的车辆数--预测每个edge的流量
        future_edge_task_nums = np.zeros([self.num_edge, self.pred_len])
        # 每个edge覆盖的车辆数--统计每个edge的流量
        edge_task_nums = [0] * self.num_edge
        # 每个edge覆盖的任务数据总量
        edge_task_data_size = [0] * self.num_edge
        # 每个edge覆盖的任务cpu_cycle总数
        edge_task_CPU_cycle = [0] * self.num_edge

        uav_obs = {}
        uav_share_obs = {}
        edge_obs = {}
        edge_share_obs = {}
        edge_next_obs = {}
        edge_next_share_obs = {}

        for id_i, edge_i in self.edges.items():
            if id_i in edge_vehicle_map.keys():
                agent_id = self.Edge_index_map[id_i]
                for uav_id, uav_i in self.uavs.items():
                    if agent_id in uav_i.covered_edges:
                        break
                edge_next_obs[agent_id] = np.zeros([len(edge_vehicle_map[id_i]), self.obs_edge_dim[agent_id]-self.num_uav_cover[uav_id]])
                edge_next_share_obs[agent_id] = np.zeros([len(edge_vehicle_map[id_i]), self.share_obs_edge_dim[agent_id]-self.num_uav_cover[uav_id]])

        whole_edge_state = []
        for _, edge_i in self.edges.items():
            whole_edge_state.append([edge_i.cpu, edge_i.core, edge_i.buff_CPU, edge_i.bw, edge_i.trans])
        whole_edge_state = np.array(whole_edge_state).flatten()

        # 对于每一个edge
        for id_i, edge_i in self.edges.items():
            agent_id = self.Edge_index_map[id_i]
            # 车辆根据模拟数据加入vehicle_pool并与Edges连接，更新连接车辆的轨迹预测值
            # edge_vehicle_map --- {edge_0:[v_1,v_2,...., ],edge_1:[],...,edge_num:[]}
            edge_i.clear_futuretrajectory()
            if id_i in next_edge_vehicle_map:
                vehicle_ids = next_edge_vehicle_map[id_i]
                for v_id in vehicle_ids:
                    if v_id not in self.vehicle_pool:  # 首次出现的车辆
                        vehicle_i = Vehicle(v_id)  # 新建一个车辆
                        self.vehicle_pool[v_id] = vehicle_i  # 加入系统的车辆pool中
                    if next_future_trajectory[v_id] != None:
                        edge_i.update_futuretrajectory(
                            np.array([self.Edge_index_map[str(int(item)) if isinstance(item, float) else item]
                                      for item in next_future_trajectory[v_id]]))  # 更新预测值
                    else:
                        edge_i.update_futuretrajectory(np.full(self.pred_len, -1))
                all_task_num += len(vehicle_ids)
                edge_task_nums[agent_id] = len(vehicle_ids)  # 统计每个edge的流量
                edge_i.vehicle_connection(vehicle_ids)  # ID 记录至对应edge
            else:
                edge_i.connected_vehicles = []

            # 每个车辆生成任务
            for v_id in edge_i.connected_vehicles:
                v_i = self.vehicle_pool[v_id]
                v_i.generate_task(self.task_pool)

            # Edge Information
            edge_state = np.array([edge_i.cpu, edge_i.core, edge_i.buff_CPU, edge_i.bw, edge_i.trans])
            # UAV Information
            for _, uav_i in self.uavs.items():
                if agent_id in uav_i.covered_edges:
                    uav_state = np.array([uav_i.cpu, uav_i.core, uav_i.buff_CPU, uav_i.bw])
                    break

            if len(edge_i.connected_vehicles) > 0:
                edge_obs[agent_id] = []
                edge_share_obs[agent_id] = []
                # Vehicle information + Task information
                for i, v_id in enumerate(edge_i.connected_vehicles):
                    v_i = self.vehicle_pool[v_id]
                    edge_task_data_size[agent_id] += v_i.current_task.data_size
                    edge_task_CPU_cycle[agent_id] += v_i.current_task.CPU_cycle
                    # [车辆算力，传输功率；任务数据量，任务所需CPU，任务截止时间]
                    vehicle_task_state = [v_i.cpu, v_i.core, v_i.buff_CPU, v_i.trans,
                                          v_i.current_task.data_size, v_i.current_task.CPU_cycle, v_i.current_task.ddl]
                    prediction_state = edge_i.predict_trajectory[i]
                    edge_obs[agent_id].append(np.hstack((vehicle_task_state, prediction_state, edge_state, uav_state)))
                    edge_share_obs[agent_id].append(np.hstack((vehicle_task_state, prediction_state, whole_edge_state, uav_state)))  # uav_state!

                    for j in edge_vehicle_map.keys():
                        if v_id in edge_vehicle_map[j]:
                            for _, uav_j in self.uavs.items():
                                if self.Edge_index_map[j] in uav_j.covered_edges:
                                    break
                            edge_next_obs[self.Edge_index_map[j]][edge_vehicle_map[j].index(v_id)] = np.hstack(
                                (vehicle_task_state, prediction_state,
                                 np.array([self.edges[j].cpu, self.edges[j].core, self.edges[j].buff_CPU, self.edges[j].bw, self.edges[j].trans]),
                                 np.array([uav_j.cpu, uav_j.core, uav_j.buff_CPU, uav_j.bw])))
                            edge_next_share_obs[self.Edge_index_map[j]][edge_vehicle_map[j].index(v_id)] = np.hstack(
                                (vehicle_task_state, prediction_state, whole_edge_state, np.array([uav_j.cpu, uav_j.core, uav_j.buff_CPU, uav_j.bw])))

                edge_obs[agent_id] = np.array(edge_obs[agent_id])
                edge_share_obs[agent_id] = np.array(edge_share_obs[agent_id])

            # 统计每个edge的未来流量
            if edge_i.predict_trajectory != []:
                edge_predict_trajectories = np.array(edge_i.predict_trajectory)
                future_edge_task_nums += np.apply_along_axis(lambda x: np.bincount(x, minlength=self.num_edge + 1),
                    axis=0, arr=np.where(edge_predict_trajectories == -1, self.num_edge, edge_predict_trajectories))[:-1, :]

        whole_edge_state = []
        whole_edge_prediction_state = []
        for id_i, edge_i in self.edges.items():
            agent_id = self.Edge_index_map[id_i]
            whole_edge_state.append([edge_i.cpu, edge_i.core, edge_i.buff_CPU, edge_i.bw, edge_i.trans, edge_task_nums[agent_id],
                                     edge_task_data_size[agent_id], edge_task_CPU_cycle[agent_id]])
            whole_edge_prediction_state.append(future_edge_task_nums[agent_id])
        whole_edge_state = np.array(whole_edge_state).flatten()
        whole_edge_prediction_state = np.array(whole_edge_prediction_state).flatten()
        whole_edge_state = np.hstack((np.array(whole_edge_state), np.array(whole_edge_prediction_state))).flatten()

        # 对于每一个UAV
        for uav_id, uav_i in self.uavs.items():
            # UAV Information
            uav_state = np.array([uav_i.cpu, uav_i.core, uav_i.buff_CPU, uav_i.bw])
            edge_state = []
            edge_prediction_state = []
            for agent_id in uav_i.covered_edges:
                for id_i, edge_i in self.edges.items():
                    if self.Edge_index_map[id_i] == agent_id:
                        break
                edge_state.append([edge_i.cpu, edge_i.core, edge_i.buff_CPU, edge_i.bw, edge_i.trans, edge_task_nums[agent_id],
                                   edge_task_data_size[agent_id], edge_task_CPU_cycle[agent_id]])
                edge_prediction_state.append(future_edge_task_nums[agent_id])
            edge_state = np.hstack((np.array(edge_state), np.array(edge_prediction_state))).flatten()
            uav_obs[uav_id] = np.hstack((edge_state, uav_state))
            uav_share_obs[uav_id] = np.hstack((whole_edge_state, uav_state))

        info = {}
        info["latencies"] = latencies
        info["wait_latencies"] = wait_latencies
        info["exe_latencies"] = exe_latencies
        info["tran_latencies"] = tran_latencies
        info["mig_latencies"] = mig_latencies
        info["energies"] = energies
        info["store_energies"] = store_energies
        info["exe_energies"] = exe_energies
        info["tran_energies"] = tran_energies
        info["mig_energies"] = mig_energies
        info["failure_nums"] = failure_nums
        info["local_nums"] = local_nums
        info["average_vehicle_utilization"] = np.mean(vehicle_resource_utilization)
        info["average_edge_utilization"] = np.mean(edge_resource_utilization)
        info["average_uav_utilization"] = np.mean(uav_resource_utilization)

        self.count_step += 1
        if self.count_step >= self.time_slots:
            self.done = True

        return all_task_num, edge_task_nums, edge_comp_task_nums, uav_comp_task_nums,\
               uav_obs, uav_share_obs, uav_rewards,\
               edge_obs, edge_share_obs, edge_next_obs, edge_next_share_obs, edge_rewards, info, self.done
