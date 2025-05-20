# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import omnisafe
from omnisafe.utils.tools import custom_cfgs_to_dict, update_dict
from envs.hidden_parameter_env import HiddenParamEnvs
# We need to import envs to register the environments
import envs

def dataclass_to_dict(config: object) -> dict:
    return {k.lower(): getattr(config, k) for k in dir(config) if not k.startswith('_')}

if __name__ == '__main__':
    from configuration import DynamicPredictorConfig, EnvironmentConfig
    # python run.py --algo ShieldedTRPOLag --env-id SafetyPointGoal1-v1 --fe-representation True
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algo',
        type=str,
        metavar='ALGO',
        default='ShieldedPPOLag',
        help='algorithm to train',
        choices=omnisafe.ALGORITHMS['all'],
    )
    parser.add_argument(
        '--env-id',
        type=str,
        metavar='ENV',
        default='SafetyPointGoal2-v1',
        help='the name of test environment',
    )
    parser.add_argument(
        '--parallel',
        default=1,
        type=int,
        metavar='N',
        help='number of paralleled progress for calculations.',
    )
    parser.add_argument(
        '--total-steps',
        type=int,
        default=2000000,
        metavar='STEPS',
        help='total number of steps to train for algorithm',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        metavar='DEVICES',
        help='device to use for training',
    )
    parser.add_argument(
        '--vector-env-nums',
        type=int,
        default=4,
        metavar='VECTOR-ENV',
        help='number of vector envs to use for training',
    )
    parser.add_argument(
        '--torch-threads',
        type=int,
        default=16,
        metavar='THREADS',
        help='number of threads to use for torch',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=100,
        metavar='SEED',
        help='random seed',
    )

    parser.add_argument(
        '--sampling-nbr',
        type=int,
        default=10,
        metavar='SAMPLING-NBR',
        help='number of samples for sampling',
    )

    parser.add_argument(    
        '--enhanced-safety',
        type=float,
        default=1.5,
        metavar='ENHANCED-SAFETY',
        help='enhanced safety',
    )

    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.0,
        metavar='ENTROPY-COEF',
        help='entropy coef',
    )

    parser.add_argument(
        '--safety-bonus',
        type=float,
        default=1.,
        metavar='SAFETY-BONUS',
        help='safety bonus',
    )

    parser.add_argument(
        '--static-threshold',
        type=float,
        default=0.275,
        metavar='STATIC-THRESHOLD',
        help='static threshold',
    )

    parser.add_argument(
        '--penalty-type',
        type=str,
        default='reward',
        metavar='PENALTY-TYPE',
        help='penalty type',
    )
    parser.add_argument(
        '--oracle',
        type=bool,
        default=False,
        metavar='ORACLE',
        help='oracle',
    )

    parser.add_argument(
        '--use-wandb',
        type=bool,
        default=False,
        metavar='USE-WANDB',
        help='whether to use wandb',
    )

    parser.add_argument(
        '--fe-representation',
        type=bool,
        default=False,
        metavar='FE-REPRESENTATION',
        help='fe representation',
    )

    parser.add_argument(
        '--use-acp',
        type=bool,
        default=False,
        metavar='USE-ACP',
        help='use acp',
    )

    parser.add_argument(
        '--steps-per-epoch',
        type=int,
        default=20000,
        metavar='STEPS-PER-EPOCH',
        help='steps per epoch',
    )

    parser.add_argument(
        '--warm-up-epochs',
        type=int,
        default=20,
        metavar='WARM-UP-EPOCHS',
        help='warm up epochs',
    )

    # batch size pareser
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        metavar='BATCH-SIZE',
        help='batch size',
    )

    # lambda init
    parser.add_argument(
        '--lagrangian_multiplier_init',
        type=float,
        default=0.001,
        metavar='LAMBDA-INIT',
        help='lambda init',
    )
    
    # lambda lr 
    parser.add_argument(
        '--lambda-lr',
        type=float,
        default=0.035,
        metavar='LAMBDA-LR',
        help='lambda lr',
    )

    parser.add_argument(
        '--prediction-horizon',
        type=int,
        default=1,
        metavar='PREDICTION-HORIZON',
        help='prediction horizon',
    )

    parser.add_argument(
        '--safety_measure_discount',
        type=float,
        default=0.9,
        metavar='SAFETY-MEASURE-DISCOUNT',
        help='safety measure discount',
    )

    parser.add_argument(
        '--project-name',
        type=str,
        default='shield',
        metavar='PROJECT-NAME',
        help='project name',
    )

    args, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    oracle = bool(vars(args).pop('oracle'))
    fe_representation = bool(vars(args).pop('fe_representation'))
    use_acp = bool(vars(args).pop('use_acp'))
    if not fe_representation:
        print("No fe representation is provided, using oracle representation instead")
        oracle = True
    assert sum([oracle, fe_representation]) == 1, "Only one of oracle or fe_representation can be True"
    env_cfgs, dynamic_predictor_cfgs = EnvironmentConfig(), DynamicPredictorConfig()
    assert env_cfgs.MIN_MULT > 0 and env_cfgs.MAX_MULT > 0, "For training, we have to manually set the min and max mult to 0.25 and 1.75 respectively in configuration.py file"
    
    env_cfgs.USE_ORACLE = oracle
    env_cfgs.USE_FE_REPRESENTATION = fe_representation
    env_cfgs.ENV_INFO = args.env_id.split('-')[0]
    
    if 'Circle' in args.env_id:
        env_cfgs.GENERAL_ENV = False
        env_cfgs.CIRCLE_ENV = True
    else:
        env_cfgs.RANGE_LIMIT = -1.0
        env_cfgs.GENERAL_ENV = True
        env_cfgs.CIRCLE_ENV = False
        
    env = HiddenParamEnvs(args.env_id, device=args.device, env_config=env_cfgs, num_envs=1)
    obs, info = env.reset()
    if 'sigwalls_loc' in info:
        env_cfgs.RANGE_LIMIT = info['sigwalls_loc'][0]

    custom_cfgs = {}
    for k, v in unparsed_args.items():
        update_dict(custom_cfgs, custom_cfgs_to_dict(k, v))

    # We should convert the dataclass to dict for the omnisafe config
    dynamic_predictor_cfgs = dataclass_to_dict(dynamic_predictor_cfgs)
    env_cfgs = dataclass_to_dict(env_cfgs)
    
    # project name
    project_name = vars(args).pop('project_name')
    
    custom_cfgs = {
        'seed': int(vars(args).pop('seed')),
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
            'steps_per_epoch': int(vars(args).pop('steps_per_epoch')),
            'batch_size': int(vars(args).pop('batch_size')),
            'entropy_coef': float(vars(args).pop('entropy_coef')),
        },
        'lagrange_cfgs': {
            'lambda_lr': vars(args).pop('lambda_lr'),
            'lagrangian_multiplier_init': vars(args).pop('lagrangian_multiplier_init'),
        }
    }

    safety_bonus = float(vars(args).pop('safety_bonus'))
    sampling_nbr = int(vars(args).pop('sampling_nbr'))
    static_threshold = float(vars(args).pop('static_threshold'))
    enhanced_safety = float(vars(args).pop('enhanced_safety'))
    penalty_type = vars(args).pop('penalty_type')
    warm_up_epochs = int(vars(args).pop('warm_up_epochs'))
    prediction_horizon = int(vars(args).pop('prediction_horizon'))
    safety_measure_discount = float(vars(args).pop('safety_measure_discount'))
    # prediction horizon 0 means that we only use SRO, shielding will not be triggered.
    assert prediction_horizon <= 1, "Since we trained the dynamics predictor to predict 'position' information only, current prediction horizon is 1 or 0"
    
    custom_cfgs['shield_cfgs'] = {
        'sampling_nbr': sampling_nbr,
        'static_threshold': static_threshold,
        'enhanced_safety': enhanced_safety,
        'safety_bonus': safety_bonus,
        'penalty_type': penalty_type,
        'warm_up_epochs': warm_up_epochs,
        'prediction_horizon': prediction_horizon,
        'safety_measure_discount': safety_measure_discount,
        'env_config': env_cfgs,
        'dynamic_predictor_cfgs': dynamic_predictor_cfgs,
        'use_acp': use_acp,
    }
    
    if 'CPO' in args.algo:
        custom_cfgs.pop('lagrange_cfgs')
    
    if not args.algo in ['CPO', 'TRPOLag', 'PPOLag', 'ShieldedTRPOLag', 'ShieldedPPOLag']:
        custom_cfgs.pop('shield_cfgs')
        custom_cfgs.pop('lagrange_cfgs')
    
    agent = omnisafe.Agent(
        args.algo,
        args.env_id,
        train_terminal_cfgs=vars(args),
        custom_cfgs=custom_cfgs,
    )

    agent.learn()
