import os
import sys
import envs
from envs.safety_gymnasium.wrappers.gymnasium_conversion import SafetyGymnasium2Gymnasium
from configuration import EnvironmentConfig
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

# This file is to train a policy to collect transition dynamics
env_config = EnvironmentConfig()
point_tasks = ['SafetyPointPush1-v1', 'SafetyPointButton1-v1', 'SafetyPointCircle1-v1', 'SafetyPointGoal1-v1']
car_tasks = ['SafetyCarPush1-v1', 'SafetyCarButton1-v1', 'SafetyCarCircle1-v1', 'SafetyCarGoal1-v1']
env_id = sys.argv[1]
total_timesteps = int(sys.argv[2])
make_env = lambda: SafetyGymnasium2Gymnasium(envs.make(env_id[:-1] + str(0), env_config=env_config))
seed = 0
set_random_seed(seed)
# Parallel environments
vec_env = make_vec_env(make_env, n_envs=4)

log_dir = f"trained_policies_for_collection/{env_id}/"
os.makedirs(log_dir, exist_ok=True)
# train for some number of steps
model = PPO("MlpPolicy", vec_env, verbose=0)
model.learn(total_timesteps=total_timesteps, progress_bar=True)
model.save(os.path.join(log_dir, "ppo_policy"))