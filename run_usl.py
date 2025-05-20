import os,argparse,warnings
warnings.filterwarnings("ignore")
from typing import Any, ClassVar, Union, Tuple, List, Dict, Optional
import pandas as pd
import numpy as np
import torch
import random
import saferl_kit.saferl_algos as saferl_algos
from saferl_kit.saferl_plotter.logger import SafeLogger
import saferl_kit.saferl_utils as saferl_utils
import envs
from envs import make
from gymnasium import spaces
from configuration import EnvironmentConfig


class HiddenParameterEnv:
    _support_envs: ClassVar[List[str]] = ['SafetyPointGoal1-v1', 'SafetyPointGoal2-v1',
                                          'SafetyCarGoal1-v1', 'SafetyCarGoal2-v1',
                                          'SafetyPointButton1-v1', 'SafetyPointButton2-v1',
                                          'SafetyCarButton1-v1', 'SafetyCarButton2-v1',
                                          'SafetyPointCircle1-v1', 'SafetyPointCircle2-v1',
                                          'SafetyCarCircle1-v1', 'SafetyCarCircle2-v1',
                                          'SafetyCarPush1-v1', 'SafetyCarPush2-v1',
                                          'SafetyPointPush1-v1', 'SafetyPointPush2-v1',
                                          ]

    def __init__(self, env_id: str, device: str, env_config: EnvironmentConfig, num_envs: int = 1, render_mode: Optional[str] = None, **kwargs) -> None:
        self.render_mode = render_mode
        self.nbr_of_gremlins = env_config.NBR_OF_GREMLINS
        self.nbr_of_static_obstacles = env_config.NBR_OF_STATIC_OBSTACLES
        self.nbr_of_goals = env_config.NBR_OF_GOALS
        self.total_objects = env_config.NBR_OF_GREMLINS + env_config.NBR_OF_STATIC_OBSTACLES
        self.env = make(env_id[:-1] + str(0), env_config=env_config, render_mode=render_mode)
        self._device = torch.device(device)
        self._count = 0
        self._num_envs = num_envs
        self.oracle = env_config.USE_ORACLE

        self._observation_space = self.env.observation_space
        bounds = {
            'original': (self._observation_space.low, self._observation_space.high),
            'hidden': ([-2.0, -2.0], [2.0, 2.0]),
        }
        low = np.concatenate([b[0] for b in bounds.values()])
        high = np.concatenate([b[1] for b in bounds.values()])
        self.observation_space = spaces.Box(low, high)
        self.action_space = self.env.action_space

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

    def reset(
        self,
        seed: Union[int, None] = None,
        options: Union[Dict[str, Any], None] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if seed is not None:
            self.set_seed(seed)

        obs, info = self.env.reset(seed=seed)
        obs = self.augment_state(obs, info)
        
        self._count = 0
        return obs, info
    
    def augment_state(self, obs: np.ndarray, info: np.ndarray) -> np.ndarray:
        """Augment state with coefficients and step counter."""
        hidden_info = info['hidden_parameters'] if self.oracle else np.zeros(2)
        return np.concatenate((obs, hidden_info))
    
    @property
    def max_episode_steps(self) -> None:
        """The max steps per episode."""
        if 'Circle' in self.env.spec.id or 'Run' in self.env.spec.id:
            return 500
        else:
            return 1000

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()
    
    def step(
        self,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self._count += 1
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info['cost'] = cost
        obs = self.augment_state(obs, info)

        # Handle final observation
        if np.any(terminated) or np.any(truncated):
            # Convert to boolean before OR operation
            info['_final_observation'] = (terminated > 0) | (truncated > 0)
            info['final_observation'] = obs
        return obs, reward, cost, terminated, truncated, info

def process_state(state):
    """
    Process state observation to ensure consistent numpy array format.
    
    Args:
        state: Raw state observation from environment
        
    Returns:
        np.ndarray: Processed state as a flat numpy array
    """
    # Convert to numpy array if not already
    if not isinstance(state, np.ndarray):
        state = np.array(state, dtype=np.float32)
    
    # Ensure the array is flat
    return state.flatten()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name",type=str)
    parser.add_argument("--env", default="SafetyPointGoal1-v1")           # Env name
    parser.add_argument("--flag", default="cost")   # c_t = info[flag]
    parser.add_argument("--base_policy", default="TD3")             # Base Policy name
    parser.add_argument("--use_td3", action="store_true")           # unconstrained RL
    parser.add_argument("--use_usl", action="store_true")           # Wether to use Unrolling Safety Layer
    parser.add_argument("--use_qpsl",action="store_true")           # Wether to use QP Safety Layer (Dalal 2018)
    parser.add_argument("--use_recovery",action="store_true")       # Wether to use Recovery RL     (Thananjeyan 2021)
    parser.add_argument("--use_lag",action="store_true")            # Wether to use Lagrangian Relaxation  (Ray 2019)
    parser.add_argument("--use_fac",action="store_true")            # Wether to use FAC (Ma 2021)
    parser.add_argument("--use_rs",action="store_true")             # Wether to use Reward Shaping
    parser.add_argument("--oracle",action="store_true")             # Wether to use Reward Shaping
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    # Hyper-parameters for all safety-aware algorithms
    parser.add_argument("--delta",default = 0.1,type=float)         # Qc(s,a) \leq \delta
    parser.add_argument("--cost_discount", default=0.99)            # Discount factor for cost-return
    # Hyper-parameters for using Safety++
    parser.add_argument("--warmup_ratio", default=1/5)              # Start using USL in traing after max_timesteps*warmup_ratio steps
    parser.add_argument("--kappa",default = 5, type=float)                      # Penalized factor for Safety++
    parser.add_argument("--early_stopping", action="store_true")    # Wether to terminate an episode upon cost > 0
    # Hyper-parameters for using Reward Shaping
    parser.add_argument("--cost_penalty",default = 0.5, type=float)               # Step-size of multiplier update
    # Hyper-parameters for using Lagrangain Relaxation
    parser.add_argument("--lam_init", default = 0.)                 # Initalize lagrangian multiplier
    parser.add_argument("--lam_lr",default = 1e-5)                # Step-size of multiplier update
    # Other hyper-parameters for original TD3
    parser.add_argument("--start_timesteps", default=5000, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5000, type=int)      # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=2e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--rew_discount", default=0.99)             # Discount factor for reward-return
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--eval_only", action="store_true")            # Model load file path
    args = parser.parse_args()

    assert [bool(i) for i in [args.use_td3,args.use_usl,args.use_recovery,args.use_qpsl,args.use_lag,args.use_fac,args.use_rs]].count(True) == 1, 'Only one option can be True'

    logdir = './logs/oracle' if args.oracle else './logs/no_oracle'
    if not args.eval_only:

        if not args.exp_name:
            if args.use_usl:
                file_name = f"{'usl'}_{args.base_policy}_{args.env}_{args.seed}"
                logger = SafeLogger(log_dir=logdir,exp_name='1_USL',env_name=args.env,seed=int(args.seed),fieldnames=['EpRet','EpCost','CostRate'])
            elif args.use_recovery:
                file_name = f"{'rec'}_{args.base_policy}_{args.env}_{args.seed}"
                logger = SafeLogger(log_dir=logdir,exp_name='2_REC',env_name=args.env,seed=int(args.seed),fieldnames=['EpRet','EpCost','CostRate'])
            elif args.use_qpsl:
                file_name = f"{'qpsl'}_{args.base_policy}_{args.env}_{args.seed}"
                logger = SafeLogger(log_dir=logdir,exp_name='3_QPSL',env_name=args.env,seed=int(args.seed),fieldnames=['EpRet','EpCost','CostRate'])
            elif args.use_lag:
                file_name = f"{'lag'}_{args.base_policy}_{args.env}_{args.seed}"
                logger = SafeLogger(log_dir=logdir,exp_name='4_LAG',env_name=args.env,seed=int(args.seed),fieldnames=['EpRet','EpCost','CostRate'])
            elif args.use_fac:
                file_name = f"{'fac'}_{args.base_policy}_{args.env}_{args.seed}"
                logger = SafeLogger(log_dir=logdir,exp_name='5_FAC',env_name=args.env,seed=int(args.seed),fieldnames=['EpRet','EpCost','CostRate'])
            elif args.use_rs:
                file_name = f"{'rs'}_{args.base_policy}_{args.env}_{args.seed}"
                logger = SafeLogger(log_dir=logdir,exp_name='6_RS',env_name=args.env,seed=int(args.seed),fieldnames=['EpRet','EpCost','CostRate'])
            else:
                file_name = f"{'unconstrained'}_{args.base_policy}_{args.env}_{args.seed}"
                logger = SafeLogger(log_dir=logdir,exp_name='7_TD3',env_name=args.env,seed=int(args.seed),fieldnames=['EpRet','EpCost','CostRate'])
        else:
            file_name = args.exp_name
            logger = SafeLogger(exp_name=args.exp_name,env_name=args.env,seed=int(args.seed),fieldnames=['EpRet','EpCost','CostRate'])

    env_id = args.env
    env_config = EnvironmentConfig()
    env_config.USE_ORACLE = True if args.oracle else False
    env = HiddenParameterEnv(env_id, device=torch.device("cuda"), num_envs=1, env_config=env_config, render_mode=None)
        
    # env = SafeNormalizeObservation(env)
    obs, info = env.reset()
    
    # env.seed(args.seed)
    env.action_space.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])


    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "rew_discount": args.rew_discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
    }

    kwargs_safe = {
        "cost_discount": args.cost_discount,
        "delta": args.delta,                     
    }


    if args.use_usl:
        from saferl_kit.saferl_algos.safetyplusplus import eval_policy
        kwargs.update(kwargs_safe)
        kwargs.update({'kappa':args.kappa})
        policy = saferl_algos.safetyplusplus.TD3Usl(**kwargs)
        replay_buffer = saferl_utils.CostReplayBuffer(state_dim, action_dim)
    elif args.use_recovery:
        from saferl_kit.saferl_algos.recovery import eval_policy
        kwargs.update(kwargs_safe)
        policy = saferl_algos.recovery.TD3Recovery(**kwargs)
        replay_buffer = saferl_utils.RecReplayBuffer(state_dim, action_dim)
    elif args.use_qpsl:
        from saferl_kit.saferl_algos.safetylayer import eval_policy
        kwargs.update(kwargs_safe)
        policy = saferl_algos.safetylayer.TD3Qpsl(**kwargs)
        replay_buffer = saferl_utils.SafetyLayerReplayBuffer(state_dim, action_dim)
    elif args.use_lag:
        from saferl_kit.saferl_algos.lagrangian import eval_policy
        kwargs.update(kwargs_safe)
        policy = saferl_algos.lagrangian.TD3Lag(**kwargs)
        replay_buffer = saferl_utils.CostReplayBuffer(state_dim, action_dim)
    elif args.use_fac:
        from saferl_kit.saferl_algos.fac import eval_policy
        kwargs.update(kwargs_safe)
        policy = saferl_algos.fac.TD3Fac(**kwargs)
        replay_buffer = saferl_utils.CostReplayBuffer(state_dim, action_dim)
    elif args.use_td3 or args.use_rs:
        from saferl_kit.saferl_algos.unconstrained import eval_policy,TD3
        policy = TD3(**kwargs)
        replay_buffer = saferl_utils.SimpleReplayBuffer(state_dim, action_dim)
    else:
        raise NotImplementedError    

    if args.eval_only:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # /home/mj/Project/2025/safe-rl-adaptive-shielding/logs/oracle/1_USL_SafetyPointButton1_v1-seed2-2
        env_id = args.env
        load_env_info = env_id.split('-')[0][:-1] + str(1)
        env_info = env_id.split('-')[0]
        load_model_dir = f'./logs/oracle/1_USL_{load_env_info}_v1-seed{args.seed}-2'
        
        print("Make sure the model is in the same directory as the script")
        policy.load(f"{load_model_dir}/usl_model", device)
        # this is OOD parameters
        env_config.MIN_MULT = -1
        env_config.MAX_MULT = -1
        eval_env = HiddenParameterEnv(env_id, device=device, num_envs=1, env_config=env_config, render_mode=None)
        episode_rewards, episode_costs, episode_hidden_parameters = eval_policy(policy, eval_env, int(args.seed), args.flag, eval_episodes=100, use_usl=True)
        output_dir = f"./ood_evaluation_folder/{env_info}/USL/seed{args.seed}"
        output_file_path = os.path.join(output_dir, "evaluation_results.csv")
        os.makedirs(output_dir, exist_ok=True)
        data_file_path = os.path.join(output_dir, "episode_data.npz")
        np.savez(
            data_file_path,
            rewards=np.array(episode_rewards),
            hidden_parameters=np.array(episode_hidden_parameters),
            costs=np.array(episode_costs),
        )
        results_data = {
        "Metric": [
            "Average episode reward",
            "Average episode cost",
        ],
        "Value": [
            np.mean(a=episode_rewards),
            np.mean(a=episode_costs),
        ]
        }
        results_df = pd.DataFrame(results_data)

        # Save the DataFrame to CSV
        results_df.to_csv(output_file_path, index=False)
        exit()
    else:
        eval_env = None
    
    
    state, info = env.reset()
    terminated = False  # Track terminated state
    truncated = False   # Track truncated state
    episode_reward = 0
    episode_cost = 0
    episode_timesteps = 0
    episode_num = 0
    cost_total = 0
    prev_cost = 0
    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        if args.use_usl:
            if t < args.start_timesteps:
                action = env.action_space.sample()
            elif t < int(args.max_timesteps * args.warmup_ratio):
                state_processed = process_state(state)
                action = policy.select_action(state_processed, use_usl=False, exploration=True)
            else:
                state_processed = process_state(state)
                action = policy.select_action(state_processed, use_usl=True, exploration=True)
        elif args.use_recovery:
            if t < args.start_timesteps:
                raw_action = env.action_space.sample()
                action = raw_action
            elif t < int(args.max_timesteps * args.warmup_ratio):
                action, raw_action = policy.select_action(process_state(state), recovery=False, exploration=True)
            else:
                action, raw_action = policy.select_action(process_state(state), recovery=True, exploration=True)
        elif args.use_qpsl:
            if t < args.start_timesteps:
                action = env.action_space.sample()
            elif t < int(args.max_timesteps * args.warmup_ratio):
                action = policy.select_action(process_state(state), use_qpsl=False, exploration=True)
            else:
                action = policy.select_action(process_state(state), use_qpsl=True, prev_cost=prev_cost, exploration=True)
        else:
            if t < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = policy.select_action(process_state(state), exploration=True)

    
        # Perform action
        next_state, reward, cost, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Combine termination conditions

        # if reward shaping
        if args.use_rs:
            reward -= args.cost_penalty * cost

        if cost > 0:
            cost_total += 1
            if args.early_stopping:
                terminated = True
                done = True

        done_bool = float(done) if episode_timesteps < env.max_episode_steps else 0

        # set the early broken state as 'cost = 1'
        if done and episode_timesteps < env.max_episode_steps:
            cost = 1

        # Store data in replay buffer
        if args.use_td3 or args.use_rs:
            replay_buffer.add(state, action, next_state, reward, done_bool)
        elif args.use_recovery:
            replay_buffer.add(state, raw_action, action, next_state, reward, cost, done_bool)
        elif args.use_qpsl:
            replay_buffer.add(state, action, next_state, reward, cost, prev_cost, done_bool)
        else:
            replay_buffer.add(state, action, next_state, reward, cost, done_bool)
            

        state = next_state
        prev_cost = cost
        episode_reward += reward
        episode_cost += cost

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done: 
            if args.use_lag:
                print(f'Lambda : {policy.lam}')
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Cost: {episode_cost:.3f}")
            logger.update([episode_reward,episode_cost,1.0*cost_total/t], total_steps=t+1)
            # Reset environment
            state, info = env.reset()  # Get info from reset
            terminated = False
            truncated = False
            done = False
            episode_reward = 0
            episode_cost = 0
            episode_timesteps = 0
            episode_num += 1
            prev_cost = 0

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            if eval_env is not None:
                if args.use_usl:
                    evalEpRet,evalEpCost = eval_policy(policy, eval_env, int(args.seed), args.flag, use_usl=True)
                elif args.use_recovery:
                    evalEpRet,evalEpCost = eval_policy(policy, eval_env, int(args.seed), args.flag, use_recovery=True)
                elif args.use_qpsl:
                    evalEpRet,evalEpCost = eval_policy(policy, eval_env, int(args.seed), args.flag, use_qpsl=True)
                else:
                    state_eval, info_eval = eval_env.reset()
                    state_eval = process_state(state_eval)  # Process initial state
                    evalEpRet,evalEpCost = eval_policy(policy, eval_env, int(args.seed), args.flag)
                
            if args.save_model:
                policy.save(f"./{log_dir}/usl_model")