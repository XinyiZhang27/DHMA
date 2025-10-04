import sys
import os
import setproctitle
from pathlib import Path
import torch

project_dir = os.path.abspath('.')
sys.path.append(project_dir)
sys.path.append(os.path.join(project_dir, "st_prediction"))

from config import get_config
from myutils.util import load_edge_ids, load_pkl
from envs.env_wrapper import Env
from runner.separated.env_runner import EnvRunner as Runner


def make_train_env(all_args):
    Edge_index_map = {
        "3-3-grid": {"J0": 0, "J1": 1, "J2": 2, "J3": 3, "J4": 4, "J5": 5, "J6": 6, "J7": 7, "J8": 8},
        "Net4": {'J11': 0, 'J12': 1, 'J0': 2, 'J2': 3, 'J3': 4, 'J17': 5, 'J6': 6, 'J8': 7, 'J9': 8, 'J14': 9,
                 'J7': 10, 'J10': 11, 'J15': 12},
        "bologna_pasubio": {"4": 0, '7': 1, '12': 2, '19': 3, '18': 4, '0': 5, '9': 6, '27': 7, '23': 8,
                            '2': 9, '29': 10, '33': 11, '32': 12, '1': 13, '15': 14, '40': 15, '39': 16,
                            '36': 17},
        "bologna_acosta": {'78': 0, '9': 1, '15': 2, '52': 3, '79': 4, '3': 5, '42': 6, '82': 7,
                           '62': 8, '68': 9, '27': 10, '12': 11, '32': 12, '50': 13, '48': 14,
                           '34': 15}
    }
    UAV_Edge_map = {
        "3-3-grid": {0: [2, 3, 8], 1: [1, 0, 5, 4], 2: [7, 6]},
        "Net4": {0: [0, 2], 1: [1, 3, 4], 2: [5, 6, 9, 10], 3: [7, 8, 11, 12]},
        "bologna_pasubio": {0: [0, 2, 1, 3], 1: [6, 7, 10, 11], 2: [14, 15], 3: [4, 8, 5, 9], 4: [12, 16, 13, 17]},
        "bologna_acosta": {0: [1, 2, 6, 7], 1: [3, 4, 8, 9], 2: [0, 5, 10, 15], 3: [14, 11, 12, 13]}
    }
    num_UAV_cover = {
        "3-3-grid": [3, 4, 2],
        "Net4": [2, 3, 4, 4],
        "bologna_pasubio": [4, 4, 2, 4, 4],
        "bologna_acosta": [4, 4, 4, 4]
    }
    edge_index_map = Edge_index_map[all_args.simulation_scenario]
    uav_edge_map = UAV_Edge_map[all_args.simulation_scenario]
    num_uav_cover = num_UAV_cover[all_args.simulation_scenario]
    Edge_IDs = load_edge_ids("sumo/data/{}/edge_position.csv".format(all_args.simulation_scenario))
    shortest_paths = load_pkl("sumo/data/{}/shortest_paths.pkl".format(all_args.simulation_scenario))
    envs = Env(all_args, Edge_IDs, edge_index_map, uav_edge_map, shortest_paths, num_uav_cover, all_args.time_slot)
    # envs.seed(all_args.seed)
    return envs, uav_edge_map, num_uav_cover


def parse_args(args, parser):
    # "3-3-grid"/"Net4"/"bologna_pasubio"/"bologna_acosta"
    parser.add_argument("--simulation_scenario", type=str, default="3-3-grid")
    parser.add_argument("--time_range", type=str, default="10h")  # "10h"/"24h"
    parser.add_argument("--time_slot", type=int, default=60)
    parser.add_argument("--num_edge", type=int, default=9)  # 9/13/18/16
    parser.add_argument("--num_uav", type=int, default=3)  # 3/4/5/4
    parser.add_argument("--train_start_epi", type=int, default=0)  # 0/0/0/0
    parser.add_argument("--train_end_epi", type=int, default=599)  # 599/1439/1439/1439
    parser.add_argument("--test_start_epi", type=int, default=150)  # 150/200,1200,620/130,855,725,500/120,1250,490;700
    parser.add_argument("--test_end_epi", type=int, default=350)  # 350/300,1300,720/230,955,825,600/220,1350,590;900
    parser.add_argument("--run_num0", type=str, default="run1")
    parser.add_argument("--run_num1", type=str, default="run1")

    # fine-tune
    parser.add_argument("--cpu_uav", type=int, default=6)  # 6/4/8/8/10
    parser.add_argument("--core_uav", type=int, default=4)  # 4/2/6/6/8
    parser.add_argument("--cpu_rsu", type=int, default=12)  # 12/10/15/15/18
    parser.add_argument("--core_rsu", type=int, default=10)  # 10/10/12/12/15
    parser.add_argument("--bw_uav", type=int, default=8)
    parser.add_argument("--bw_rsu", type=int, default=10)
    parser.add_argument("--trans_rate_edge", type=int, default=50)  # 50/100/100/50/100
    parser.add_argument("--coef1", type=float, default=0.5)
    parser.add_argument("--coef2", type=float, default=0.5)
    parser.add_argument("--task_parameter", type=list, default=[(60, 90), (3, 5), (3, 5)])

    parser.add_argument("--stage", type=str, default="train")  # train/test
    parser.add_argument("--strategy", type=str, default="mappo")

    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    assert (all_args.use_recurrent_policy0 == False and all_args.use_naive_recurrent_policy0 == False), "check recurrent policy!"
    assert (all_args.use_recurrent_policy1 == False and all_args.use_naive_recurrent_policy1 == False), "check recurrent policy!"

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        # torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        # torch.set_num_threads(all_args.n_training_threads)

    # run dir
    dir = (
        Path(os.path.dirname(os.path.abspath(__file__)) + "/results")
        / all_args.simulation_scenario
        / all_args.time_range
    )
    if not dir.exists():
        os.makedirs(str(dir))

    if not dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1]) for folder in dir.iterdir() if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # 测试时需分别填入训好的UAV和RSU模型的路径
    # if all_args.stage == "test":
    #     all_args.model_dir0 = dir / all_args.run_num0
    #     all_args.model_dir1 = dir / all_args.run_num1

    setproctitle.setproctitle(str(all_args.simulation_scenario))

    # seed
    # torch.manual_seed(all_args.seed)  # 设置生成随机数的主种子。确保每次运行时，使用相同的主种子生成相同的随机数，以便实验的可重复性。
    # torch.cuda.manual_seed_all(all_args.seed)  # 设置PyTorch中所有可见的CUDA设备的种子。确保在使用GPU时，生成的随机数也是可重复的。
    # np.random.seed(all_args.seed)  # 确保在使用NumPy生成的随机数也是可重复的。

    # env init
    envs, uav_edge_map, num_uav_cover = make_train_env(all_args)

    config = {
        "all_args": all_args,
        "envs": envs,
        "uav_edge_map": uav_edge_map,
        "num_uav_cover": num_uav_cover,
        "device": device,
        "run_dir": run_dir,
    }

    all_args.stage = "train"
    all_args.strategy = "mappo"
    runner = Runner(config)
    if all_args.stage == "train":
        if all_args.strategy == "mappo":
            runner.run(all_args.train_start_epi, all_args.train_end_epi)
    elif all_args.stage == "test":
        if all_args.strategy == "mappo":
            runner.test(all_args.test_start_epi, all_args.test_end_epi)

    # post process
    envs.close()
    runner.writer.close()

if __name__ == "__main__":
    main(sys.argv[1:])
