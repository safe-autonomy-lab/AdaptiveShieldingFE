# Adaptive Shielding for Safe Reinforcement Learning under Hidden-Parameter Dynamics Shifts

## Repository Structure
This repository includes the following key directories and files to support our safe reinforcement learning (RL) framework with function encoder representation learning:

- **OmniSafe**: We include the `omnisafe` directory to integrate function encoder representation learning with safe RL algorithms implemented by OmniSafe.
- **FunctionEncoder**: This directory contains the `FunctionEncoder` module, including the transition dataset dataclass and utilities for model saving and loading.
- **Shield**: The `shield` directory houses all supported shielded algorithms, including the shielding mechanism and Safe Reinforced Optimization (SRO).
- **Configuration Files**:
  - Baseline algorithm parameters are located in `omnisafe/configs/on-policy`.
  - Shielding algorithm parameters are located in `omnisafe/configs/shield`.


## Overview
For function encoder, check the folder `FunctionEncoder`, we directly copied the folder and modified relevant part from [FunctionEncoder](https://github.com/tyler-ingebrand/FunctionEncoder) for our usage.

For Conformal preidction, check files `./shield/conformal_prediction.py`, `./shield/base_shield.py`, and `./shield/vectorized_shield.py`.

To integrate shielding with algorithm, we need a seperate policy wrapper, like `./shield/adapter_wrapper.py` and `./shield/onpolicy_wrapper.py`. 

To use safety-regularized optimization (SRO), we need `./shield/model/constraint_actor_q_and_v_critic.py`, adding Q-value estimation. Accordingly, we also need shielded algorithm code `./shield/algorithms/` for each shielded version of RL algorithms. 

We provide a one shot script (train dynamics, collect dataset, train policy, evaluate policy) in `train.sh`. Read Usage section for more detail.


## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Download texture assets:**
   ```bash
   # For users with the zip file distribution
   # Download textures from https://github.com/PKU-Alignment/safety-gymnasium/tree/main/safety_gymnasium/assets/textures
   # Then, place in the correct directory
   mv textures envs/safety_gymnasium/assets/
   ```

## Usage

Follow these steps in order to train, shield, and evaluate your RL agent.
We provide a one-shot train + evaluation script in `train.sh` that runs the entire pipeline from data collection to evaluation in a single file.

### Quickstart (Single-File Pipeline)

Run everything (collect transitions → train dynamics → train policy → evaluate) with:
```bash
# Bash pipeline (edit variables at top of file)
bash train.sh
```

To change algorithm, environment, horizon, and other settings, edit the variables at the top of `train.sh`:
- `ENV_ID`: environment id (e.g., `SafetyHalfCheetahVelocity-v1`)
- `ALGO_NAME`: algorithm (e.g., `ShieldedRCPO`, `FOCOPS`)
- `HORIZON`: prediction horizon
- `SEED`, `N_BASIS`, `TOTAL_STEPS`, `PENALTY_TYPE`, etc.
- Evaluation grid: `SAMPLING_NBRS`, `THRESHOLDS`, `IDLE_CONDITIONS`, `SCALES`

Note: This workflow uses random policy transitions by default (`USE_TRAINED_POLICY=0`). Set it to `1` after running the optional pre-training step below.

### Quickstart (Python Pipeline)

You can also run the full workflow using the Python pipeline with flags:
```bash
# Python pipeline (flags override defaults)
python train_pipeline.py --env-id SafetyHalfCheetahVelocity-v1 --algo ShieldedRCPO --horizon 7 --seed 100
```

Both scripts skip transition collection and dynamics training automatically for non-shielded algorithms.

### 1. (Optional) Pre-train Policies for Data Collection

```bash
# Arguments: <env_id> <timesteps>
python 0.train_policies.py SafetyPointGoal1-v1 2000000
```

### 2. Collect Transitions Dataset

```bash
# Arguments: <env_id> <num_episodes> <use_trained_policy> <prediction_horizon>
# Set use_trained_policy=1 if you completed step 1, otherwise 0 for random policy.
# Note: the collector uses the base env (e.g., SafetyPointGoal1-v1 -> SafetyPointGoal1-v0) for non-velocity tasks.
python 1.collect_transition.py SafetyPointGoal1-v1 1000 0 1
```

### 3. Train Dynamics Predictor (Function Encoder / Transformer / PEM / Oracle)

```bash
# Example: train FE dynamics model for 1-step prediction (uses data from step 2)
python 2.train_dynamics_predictor.py --env_id SafetyPointGoal1-v1 --dynamics_model fe --prediction_horizon 1 --seed 0
```

### 4. Train with Adaptive Shielding

```bash
# Generic command
python run.py \
  --env-id <env_id> \
  --algo <algorithm> \
  --prediction-horizon <0|1|k> \
  --penalty-type <reward|shield> \
  --sampling-nbr <sampling_number> \
  --safety-bonus <bonus_weight> \
  --idle-condition 4 \
  --use-wandb <True|False> \
  --fe-representation <True|False> \
  --project-name <project_name>
```

**Available algorithms:**
- Shielded algorithms: `ShieldedTRPOLag`, `ShieldedPPOLag`, `ShieldedRCPO`
- Baseline algorithms: `PPOLag`, `TRPOLag`, `CUP`, `CPO`, `TRPOSaute`, `PPOSaute`, `FOCOPS`, `RCPO`, `RCPOSaute` (these use oracle representation automatically, unless it's specified)

**Example command:**
```bash
python run.py \
  --env-id SafetyPointGoal1-v1 \
  --algo ShieldedRCPO \
  --prediction-horizon 1 \
  --penalty-type reward \
  --sampling-nbr 10 \
  --safety-bonus 1. \
  --idle-condition 4 \
  --use-wandb True \
  --fe-representation True \
  --project-name shield 
```

#### Key Parameters:
- `--prediction-horizon`: positive integer for one-step or multi-step shielding (values <=0 are treated as 1 in `run.py`)
- `--penalty-type`: `reward` (use SRO), `shield` (do not use SRO during optimization)
- `--fe-representation`: `True` (function encoder adaptation), `False` (oracle adaptation)
- `--sampling-nbr`: Number of action samples when adaptive shield is triggered
- `--safety-bonus`: Weight of safety in the augmented objective
- `--idle-condition`: Control frequent Shielding trigger, letting terms between activation of the Shielding

#### Notes:
- Shielding is only applied for `Shielded*` algorithms. Use baseline algorithms to disable shielding entirely.
- `--penalty-type` controls SRO: `reward` is for SRO + Shielding; `sro` uses only SRO; `shield` disables SRO but keeps the shield.

### 5. Run Unrolling Safety Layer (USL) Baseline

```bash
# Generic command
python run_usl.py --env <env_id> --use_usl --seed <seed> --oracle --save_model

# Example command
python run_usl.py --env SafetyPointGoal1-v1 --use_usl --seed 0 --oracle --save_model
```

### 6. Evaluate OOD Generalization
After training, you can find the trained model in the generated `runs` folders.
Organized results are saved under `results/` and the algorithm folder name reflects `penalty_type`:
- `reward` -> `Shielded*withSRO`
- `sro` -> `<BaseAlgo>withSRO` (e.g., `RCPOwithSRO`)
- `shield` -> `Shielded*`

For OOD testing, use environments with level 2 (e.g., SafetyPointGoal2-v1). These environments have:
- 2 additional hazard spaces 
- Hidden parameters sampled from OOD range
- Shielding is only used for `Shielded*` algorithms; baselines ignore shield settings

Also, we can control presafety condition `threshold`, higher `threshold` value leads to more conservative shielding trigger, e.g., trigger shield in distance 10, instead of 5. Scale factor control sampling diversity, when we check future steps. Idle condition controls shielding frequency. These parameters can be used as test-time tunning where the other baselines do not have.

```bash
# Generic command
python 3.load_model.py <env_id> <algorithm> <seed> <sampling_nbr> <prediction_horizon> <threshold> <idle_condition> <scale> <num_eval_episodes> <n_basis>

# Example command
python 3.load_model.py SafetyPointGoal2-v1 ShieldedTRPO 0 100 1 0.25 4 0.05 100 4
```

### One-Line Training (pipeline sample)
Use the pipeline wrapper to run training in a single command (matches `train_pipeline.py`/`train.sh` defaults):
```bash
python train_pipeline.py --env-id SafetyPointPush1-v1 --algo ShieldedRCPO --horizon 1 --penalty-type reward --total-steps 2000000 --seed 2 --n-basis 4 --safety-bonus 1.0
```

For SLURM servers, use the provided launcher:
```bash
bash run.sh
```

After OOD evaluation finishes, aggregate Pareto-optimal results (reads all config folders, not just `Shield_*`):
```bash
python pareto_report.py --aggregate-root ood_evaluation_folder
```


## Acknowledgements

This code leverages and extends:
- [saferl_kit](https://github.com/zlr20/saferl_kit)
- [OmniSafe](https://github.com/PKU-Alignment/omnisafe)
- [Safe-Gym](https://github.com/PKU-Alignment/safety-gymnasium)
- [FunctionEncoder](https://github.com/tyler-ingebrand/FunctionEncoder)

## License

Distributed under the MIT License. See `LICENSE` for details. 
