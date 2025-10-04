import argparse


def get_config():

    parser = argparse.ArgumentParser(
        description="mappo_mappo", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # #Common parameters
    # prepare parameters
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument(
        "--cuda",
        action="store_false",
        default=True,
        help="by default True, will use GPU to train; or else will use CPU",
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_false",
        default=True,
        help="by default, make sure random seed effective. if set, bypass such function.",
    )
    """
    parser.add_argument(
        "--n_training_threads",
        type=int,
        default=1,
        help="Number of torch threads for training",
    )
    """
    # env parameter
    """
    parser.add_argument(
        "--use_obs_instead_of_state",
        action="store_true",
        default=False,
        help="Whether to use global state or concatenated obs",
    )
    """
    # run parameters
    parser.add_argument(
        "--use_linear_lr_decay",
        action="store_false",
        default=True,
        help="use a linear schedule on the learning rate",
    )
    # save parameters
    parser.add_argument(
        "--save_interval",
        type=int,
        default=50,
        help="time duration between continuous twice models saving.",
    )
    # log parameters
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="time duration between continuous twice log printing.",
    )

    # #MAPPO-UAV
    # prepare parameter
    parser.add_argument("--algorithm_name0", type=str, default="mappo")
    # replay buffer parameters
    parser.add_argument("--buffer_length0", type=int, default=30000)
    # federated learning parameters
    parser.add_argument(
        "--adaptive_num",
        type=int,
        default=2,
        help="Number of adaptive aggregated higher layers of actor/critic networks"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="Weight learning rate. Default: 1.0"
    )
    parser.add_argument(
        "--ppo_epoch_weights",
        type=int,
        default=15,
        help="Number of ppo epochs for weight learning"
    )
    parser.add_argument(
        "--num_mini_batch_weights",
        type=int,
        default=5,
        help="number of batches for ppo when learning weights (default: 5)",
    )
    parser.add_argument(
        "--data_chunk_length_weights",
        type=int,
        default=10,
        help="Time length of chunks used to train a recurrent_policy when learning weights",
    )
    # network parameters
    parser.add_argument(
        "--use_centralized_Q0",
        action="store_false",
        default=True,
        help="Whether to use centralized Q function",
    )
    parser.add_argument(
        "--stacked_frames0",
        type=int,
        default=1,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--hidden_size0",
        type=int,
        default=128,
        help="Dimension of hidden layers for actor/critic networks (default: 64)",
    )
    parser.add_argument(
        "--layer_N0",
        type=int,
        default=1,
        help="Number of layers for actor/critic networks",
    )
    parser.add_argument(
        "--use_ReLU0",
        action="store_false",
        default=True,
        help="Whether to use ReLU")
    parser.add_argument(
        "--use_feature_normalization0",
        action="store_false",
        default=True,
        help="Whether to apply layernorm to the inputs",
    )
    parser.add_argument(
        "--use_orthogonal0",
        action="store_false",
        default=True,
        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases",
    )
    parser.add_argument(
        "--gain0", type=float, default=0.01, help="The gain # of last action layer")
    parser.add_argument(
        "--use_popart0",
        action="store_true",
        default=False,
        help="by default False, use PopArt to normalize rewards.",
    )
    parser.add_argument(
        "--use_valuenorm0",
        action="store_false",
        default=True,
        help="by default True, use running mean and std to normalize rewards.",
    )
    # recurrent parameters
    parser.add_argument(
        "--use_naive_recurrent_policy0",
        action="store_true",
        default=False,
        help="Whether to use a naive recurrent policy",
    )
    parser.add_argument(
        "--use_recurrent_policy0",
        action="store_true",
        default=False,
        help="use a recurrent policy",
    )
    parser.add_argument(
        "--recurrent_N0",
        type=int,
        default=1,
        help="The number of recurrent layers.")
    parser.add_argument(
        "--data_chunk_length0",
        type=int,
        default=10,
        help="Time length of chunks used to train a recurrent_policy",
    )
    # optimizer parameters
    parser.add_argument(
        "--lr0", type=float, default=15e-5, help="learning rate (default: 5e-4)")
    parser.add_argument(
        "--opti_eps0",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument("--weight_decay0", type=float, default=0)
    # ppo parameters
    parser.add_argument(
        "--ppo_epoch0", type=int, default=15, help="number of ppo epochs (default: 15)")
    parser.add_argument(
        "--use_clipped_value_loss0",
        action="store_false",
        default=True,
        help="by default, clip loss value. If set, do not clip loss value.",
    )
    parser.add_argument(
        "--clip_param0",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--num_mini_batch0",
        type=int,
        default=5,
        help="number of batches for ppo (default: 5)",
    )
    parser.add_argument(
        "--entropy_coef0",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value_loss_coef0",
        type=float,
        default=1,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--use_max_grad_norm0",
        action="store_false",
        default=True,
        help="by default, use max norm of gradients. If set, do not use.",
    )
    parser.add_argument(
        "--max_grad_norm0",
        type=float,
        default=10.0,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument(
        "--use_gae0",
        action="store_false",
        default=True,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gamma0",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--gae_lambda0",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--use_proper_time_limits0",
        action="store_true",
        default=False,
        help="compute returns taking into account time limits",
    )
    parser.add_argument(
        "--use_huber_loss0",
        action="store_false",
        default=True,
        help="by default, use huber loss. If set, do not use huber loss.",
    )
    parser.add_argument(
        "--use_value_active_masks0",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in value loss.",
    )
    parser.add_argument(
        "--use_policy_active_masks0",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in policy loss.",
    )
    parser.add_argument(
        "--huber_delta0", type=float, default=10.0, help=" coefficience of huber loss.")
    # pretrained parameters
    parser.add_argument(
        "--model_dir0",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.",
    )

    # #MAPPO-Edge
    # prepare parameter
    parser.add_argument("--algorithm_name1", type=str, default="mappo")
    # replay buffer parameters
    parser.add_argument("--buffer_length1", type=int, default=30000)
    # network parameters
    parser.add_argument(
        "--use_centralized_Q1",
        action="store_false",
        default=True,
        help="Whether to use centralized Q function",
    )
    parser.add_argument(
        "--stacked_frames1",
        type=int,
        default=1,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--hidden_size1",
        type=int,
        default=128,
        help="Dimension of hidden layers for actor/critic networks (default: 64)",
    )
    parser.add_argument(
        "--layer_N1",
        type=int,
        default=1,
        help="Number of layers for actor/critic networks",
    )
    parser.add_argument(
        "--use_ReLU1",
        action="store_false",
        default=True,
        help="Whether to use ReLU")
    parser.add_argument(
        "--use_feature_normalization1",
        action="store_false",
        default=True,
        help="Whether to apply layernorm to the inputs",
    )
    parser.add_argument(
        "--use_orthogonal1",
        action="store_false",
        default=True,
        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases",
    )
    parser.add_argument(
        "--gain1", type=float, default=0.01, help="The gain # of last action layer")
    parser.add_argument(
        "--use_popart1",
        action="store_true",
        default=False,
        help="by default False, use PopArt to normalize rewards.",
    )
    parser.add_argument(
        "--use_valuenorm1",
        action="store_false",
        default=True,
        help="by default True, use running mean and std to normalize rewards.",
    )
    # recurrent parameters
    parser.add_argument(
        "--use_naive_recurrent_policy1",
        action="store_true",
        default=False,
        help="Whether to use a naive recurrent policy",
    )
    parser.add_argument(
        "--use_recurrent_policy1",
        action="store_true",
        default=False,
        help="use a recurrent policy",
    )
    parser.add_argument(
        "--recurrent_N1",
        type=int,
        default=1,
        help="The number of recurrent layers.")
    parser.add_argument(
        "--data_chunk_length1",
        type=int,
        default=10,
        help="Time length of chunks used to train a recurrent_policy",
    )
    # optimizer parameters
    parser.add_argument(
        "--lr1", type=float, default=15e-5, help="learning rate (default: 5e-4)")
    parser.add_argument(
        "--opti_eps1",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument("--weight_decay1", type=float, default=0)
    # ppo parameters
    parser.add_argument(
        "--ppo_epoch1", type=int, default=15, help="number of ppo epochs (default: 15)")
    parser.add_argument(
        "--use_clipped_value_loss1",
        action="store_false",
        default=True,
        help="by default, clip loss value. If set, do not clip loss value.",
    )
    parser.add_argument(
        "--clip_param1",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--num_mini_batch1",
        type=int,
        default=5,
        help="number of batches for ppo (default: 5)",
    )
    parser.add_argument(
        "--entropy_coef1",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value_loss_coef1",
        type=float,
        default=1,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--use_max_grad_norm1",
        action="store_false",
        default=True,
        help="by default, use max norm of gradients. If set, do not use.",
    )
    parser.add_argument(
        "--max_grad_norm1",
        type=float,
        default=10.0,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument(
        "--use_gae1",
        action="store_false",
        default=True,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gamma1",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--gae_lambda1",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--use_proper_time_limits1",
        action="store_true",
        default=False,
        help="compute returns taking into account time limits",
    )
    parser.add_argument(
        "--use_huber_loss1",
        action="store_false",
        default=True,
        help="by default, use huber loss. If set, do not use huber loss.",
    )
    parser.add_argument(
        "--use_value_active_masks1",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in value loss.",
    )
    parser.add_argument(
        "--use_policy_active_masks1",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in policy loss.",
    )
    parser.add_argument(
        "--huber_delta1", type=float, default=10.0, help=" coefficience of huber loss.")
    # pretrained parameters
    parser.add_argument(
        "--model_dir1",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.",
    )

    return parser
