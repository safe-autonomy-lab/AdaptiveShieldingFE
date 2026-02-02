import argparse
import omnisafe
from omnisafe.utils.tools import custom_cfgs_to_dict, update_dict
from envs.hidden_parameter_env import HiddenParamEnvs
from envs.safety_gymnasium.configuration import EnvironmentConfig
import torch
import logging
import os
from datetime import datetime
# We need to import envs to register the environments
import envs

def dataclass_to_dict(config: object) -> dict:
    return {k.lower(): getattr(config, k) for k in dir(config) if not k.startswith('_')}

if __name__ == '__main__':
    
    # python run.py --algo ShieldedTRPOLag --env-id SafetyPointGoal1-v1 --fe-representation True
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--algo', type=str, metavar='ALGO', default='ShieldedRCPO', help='algorithm to train', choices=omnisafe.ALGORITHMS['all'])
    parser.add_argument('--env-id', type=str, metavar='ENV', default='SafetySwimmerVelocity-v1', help='the name of test environment')
    parser.add_argument('--total-steps', type=int, default=2000000, metavar='STEPS', help='total number of steps to train for algorithm')
    parser.add_argument('--device', type=str, default='cuda:0', metavar='DEVICES', help='device to use for training')
    parser.add_argument('--vector-env-nums', type=int, default=2, metavar='VECTOR-ENV', help='number of vector envs to use for training')
    parser.add_argument('--seed', type=int, default=100, metavar='SEED', help='random seed')
    parser.add_argument('--sampling-nbr', type=int, default=10, metavar='SAMPLING-NBR', help='number of samples for sampling')
    parser.add_argument('--entropy-coef', type=float, default=0.01, metavar='ENTROPY-COEF', help='entropy coef')
    parser.add_argument('--safety-bonus', type=float, default=1., metavar='SAFETY-BONUS', help='safety bonus')
    parser.add_argument('--static-threshold', type=float, default=0.25, metavar='STATIC-THRESHOLD', help='static threshold')
    parser.add_argument('--penalty-type', type=str, default='reward', metavar='PENALTY-TYPE', help='penalty type. reward: shielding + sro, shield: only shielding, sro: only sro', choices=['reward', 'shield', 'sro'])
    parser.add_argument('--oracle', type=bool, default=False, metavar='ORACLE', help='oracle')
    parser.add_argument('--use-wandb', type=bool, default=False, metavar='USE-WANDB', help='whether to use wandb')
    parser.add_argument('--fe-representation', type=bool, default=False, metavar='FE-REPRESENTATION', help='fe representation')
    parser.add_argument('--steps-per-epoch', type=int, default=20000, metavar='STEPS-PER-EPOCH', help='steps per epoch')
    parser.add_argument('--batch-size', type=int, default=128, metavar='BATCH-SIZE', help='batch size')
    parser.add_argument('--lagrangian-multiplier-init', type=float, default=0.001, metavar='LAMBDA-INIT', help='lambda init')
    parser.add_argument('--lambda-lr', type=float, default=0.035, metavar='LAMBDA-LR', help='lambda lr')
    parser.add_argument('--prediction-horizon', type=int, default=1, metavar='PREDICTION-HORIZON', help='prediction horizon (0 maps to 1)')
    parser.add_argument('--project-name', type=str, default='[shield] adaptive shield', metavar='PROJECT-NAME', help='project name')
    parser.add_argument('--is-out-of-distribution', type=bool, default=False, metavar='IS-OUT-OF-DISTRIBUTION', help='is out of distribution')
    parser.add_argument('--fix-hidden-parameters', type=bool, default=False, metavar='FIX-HIDDEN-PARAMETERS', help='fix hidden parameters')
    parser.add_argument('--load-model-epoch', type=int, default=0, metavar='LOAD-MODEL-EPOCH', help='load model epoch')
    parser.add_argument('--idle-condition', type=int, default=4, metavar='IDLE-CONDITION', help='idle condition')
    parser.add_argument('--dynamics-model', type=str, default='fe', metavar='DYNAMICS-MODEL', help='dynamics model')
    parser.add_argument('--n-basis', type=int, default=16, metavar='N-BASIS', help='number of basis functions')
    
    shielded_algo = ['ShieldedPPOLag', 'ShieldedTRPOLag', 'ShieldedRCPO']
    baselines = ['PPOLag', 'TRPOLag', 'CUP', 'CPO', 'TRPOSaute', 'PPOSaute', 'FOCOPS', 'RCPO', 'RCPOSaute']
            
    args, unparsed_args = parser.parse_known_args()

    assert args.dynamics_model in ['fe', 'transformer', 'pem', 'oracle'], "Dynamics model should be one of fe, transformer, pem, or oracle"
    assert args.algo in shielded_algo + baselines, "Algorithm not supported"
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    fe_representation = bool(vars(args).pop('fe_representation'))
    if 'Shielded' in args.algo and args.dynamics_model in ['fe', 'transformer']:
        fe_representation = True
    load_model_epoch = int(vars(args).pop('load_model_epoch'))
    seed = int(vars(args).pop('seed'))
    steps_per_epoch = int(vars(args).pop('steps_per_epoch'))
    idle_condition = int(vars(args).pop('idle_condition'))
    if not fe_representation:
        print("No FE representation is provided, using oracle representation instead")
        oracle = True
    else:
        print("Using FE representation")
        oracle = False
    
    assert sum([oracle, fe_representation]) == 1, "Only one of oracle or fe_representation can be True"
    
    env_cfgs = EnvironmentConfig()
    assert env_cfgs.MIN_MULT > 0 and env_cfgs.MAX_MULT > 0, "For training, we have to manually set the min and max mult to 0.25 and 1.75 respectively in configuration.py file"
    bool(vars(args).pop('oracle'))
    env_cfgs.USE_ORACLE = oracle
    env_cfgs.ENV_ID = args.env_id
    env_cfgs.IS_OUT_OF_DISTRIBUTION = bool(vars(args).pop('is_out_of_distribution'))
    env_cfgs.FIX_HIDDEN_PARAMETERS = bool(vars(args).pop('fix_hidden_parameters'))
    n_basis = int(vars(args).pop('n_basis'))
    env_cfgs.NBR_OF_BASIS = n_basis
    if 'Velocity' in args.env_id:
        env_cfgs.MIN_MULT = 0.7
        env_cfgs.MAX_MULT = 1.3

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_name = f"shield_{args.env_id}_{args.algo}_h{args.prediction_horizon}_seed{seed}_{timestamp}.log"
    log_path = os.path.join("logs", log_name)
    os.environ["SHIELD_LOG_PATH"] = log_path
    os.makedirs("logs", exist_ok=True)
    shield_logger = logging.getLogger("shield")
    shield_logger.setLevel(logging.INFO)
    shield_logger.propagate = False
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path) for h in shield_logger.handlers):
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        file_handler.setFormatter(formatter)
        shield_logger.addHandler(file_handler)
    if args.prediction_horizon <= 0:
        print("prediction_horizon <= 0; defaulting to 1 for model loading and shield.")
        args.prediction_horizon = 1
    # env = HiddenParamEnvs(args.env_id, device=args.device, env_config=env_cfgs, num_envs=1, env_kwargs={'env_config': env_cfgs})
    
    # Test the env if it is fixed
    """
    env.reset()
    for i in range(10):
        action = env.action_space.sample()
        action = torch.from_numpy(action[None]).to(args.device)
        obs, reward, cost, terminated, truncated, info = env.step(action)
        print("Hidden parameters features: ", info['hidden_parameters_features'])
        if terminated or truncated:
            break
    exit()
    """

    custom_cfgs = {}
    for k, v in unparsed_args.items():
        update_dict(custom_cfgs, custom_cfgs_to_dict(k, v))

    # We should convert the dataclass to dict for the omnisafe config
    env_cfgs = dataclass_to_dict(env_cfgs)
    # project name
    project_name = vars(args).pop('project_name')
    
    custom_cfgs = {
        'seed': seed,
        'logger_cfgs': {
            'use_wandb': vars(args).pop('use_wandb'),
            'wandb_project': project_name,
        },
        'train_cfgs': {
            'total_steps': int(vars(args).pop('total_steps')),
        },
        'env_cfgs': {
            'env_config': env_cfgs
        },
        'algo_cfgs': {
            'steps_per_epoch': steps_per_epoch,
            'batch_size': int(vars(args).pop('batch_size')),
            'entropy_coef': float(vars(args).pop('entropy_coef')),
            'obs_normalize': True,
        },
        'lagrange_cfgs': {
            'lambda_lr': vars(args).pop('lambda_lr'),
            'lagrangian_multiplier_init': vars(args).pop('lagrangian_multiplier_init'),
        }
    }

    safety_bonus = float(vars(args).pop('safety_bonus'))
    sampling_nbr = int(vars(args).pop('sampling_nbr'))
    static_threshold = float(vars(args).pop('static_threshold'))
    penalty_type = vars(args).pop('penalty_type')
    assert penalty_type in ['reward','shield','sro'], "Penalty type should be 'reward', 'shield', or 'sro'"
    prediction_horizon = int(vars(args).pop('prediction_horizon'))
    dynamics_model = vars(args).pop('dynamics_model')
    
    
    custom_cfgs['shield_cfgs'] = {
        'sampling_nbr': sampling_nbr,
        'static_threshold': static_threshold,
        'safety_bonus': safety_bonus,
        'penalty_type': penalty_type,
        'prediction_horizon': prediction_horizon,
        'use_fe_representation': fe_representation,
        'idle_condition': idle_condition,
        'dynamics_model': dynamics_model,
        'n_basis': n_basis,
    }
    
    if args.algo in ['CPO', 'PPOSaute', 'TRPOSaute']:
        custom_cfgs.pop('lagrange_cfgs')
    
    algo = args.algo
    env_id = args.env_id
    agent = omnisafe.Agent(
        algo,   
        env_id,
        train_terminal_cfgs=vars(args),
        custom_cfgs=custom_cfgs,
    )

    # To save the time, we load the model from the SRO file
    if load_model_epoch > 0 and 'Shielded' in algo:
        model_path = f"./results/fe/{env_id}/SRO-{algo}_b1.0/seed{seed}/torch_save/epoch-{load_model_epoch}.pt"
        agent.load_model(model_path, training_steps=load_model_epoch * steps_per_epoch)

    agent.learn()
