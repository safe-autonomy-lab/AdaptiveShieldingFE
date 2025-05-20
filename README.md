# Runtime Safety through Adaptive Shielding: From Hidden Parameter Inference to Provable Guarantees

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Evaluation and Visualization](#-evaluation-and-visualization)
- [Advanced Configuration](#-advanced-configuration)
- [Supported Algorithms](#-supported-algorithms)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

## 📖 Overview

**Adaptive Shielding** is a framework that combines safety-related objectives (SRO) with learned dynamics models to actively shield RL agents from unsafe actions in environments with hidden dynamics. Our approach:

- Builds on **Constrained Hidden-Parameter MDPs**
- Uses **Function Encoders** for real-time inference of unobserved parameters
- Employs **conformal prediction** to provide probabilistic safety guarantees with minimal runtime overhead

## ✨ Features

- **Problem Addressed:** Variations in hidden parameters (e.g., a robot's mass distribution or friction) introduce safety risks at deployment time.

- **Core Components:**  
  - **Function Encoders** infer latent dynamics from recent transitions
  - **Safety-regularized objective (SRO)** incentivizes minimal safety violations during training
  - **Runtime shield** forecasts one-step safety risks and blocks unsafe actions
  - **Conformal prediction** quantifies uncertainty for probabilistic safety guarantees

- **Key Benefits:**  
  - Rigorous safety guarantees  
  - Fast online adaptation  
  - Strong out-of-distribution (OOD) generalization  
  - Minimal overhead compared to unconstrained RL  

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/safe-rl-adaptive-shield.git
   cd safe-rl-adaptive-shield
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Required Dependencies:**
   - PyTorch
   - Gymnasium
   - NumPy
   - Matplotlib (for visualization)
   - Wandb (optional, for experiment tracking)

4. **Download texture assets:**
   ```bash
   # For users with the zip file distribution
   # Download textures in this folder, from Google Drive due to supplementary material size limits
   https://drive.google.com/file/d/1AP2a_SifAwiM1CIRggGwiWqr62H25IID/view
   # Unzip the file once you donwload it in this folder
   # Then, place in the correct directory
   mv textures envs/safety_gymnasium/assets/
   ```

## 🛠️ Usage

Follow these steps in order to train, shield, and evaluate your RL agent:

### 1. (Optional) Pre-train Policies for Data Collection

```bash
# Arguments: <env_id> <timesteps>
python 0.train_policies.py SafetyPointGoal1-v1 2000000
```

### 2. Collect Transitions Dataset

```bash
# Arguments: <env_id> <num_episodes> <use_trained_policy>
# Set use_trained_policy=1 if you completed step 1, otherwise 0 for random policy
python 1.collect_transition.py SafetyPointGoal1-v1 1000 0
```

### 3. Train Function Encoder

Function encoder settings can be configured in the `configuration.py` file.

```bash
# Arguments: <env_id> <seed> <use_wandb>
python 2.compare_dynamic_predictors.py SafetyPointGoal1-v1 0 0
```

### 4. Train with Adaptive Shielding

```bash
# Generic command
python run.py \
  --env_id <env_id> \
  --algo <algorithm> \
  --prediction_horizon <0|1> \
  --penalty-type <reward|shield> \
  --sampling-nbr <sampling_number> \
  --safety-bonus <bonus_weight> \
  --use-wandb <True|False> \
  --fe-representation <True|oracle> \
  --use-acp <True|False> \
  --project-name <project_name>
```

**Available algorithms:**
- Shielded algorithms: `ShieldedTRPOLag`, `ShieldedPPOLag`
- Baseline algorithms: `PPOLag`, `TRPOLag`, `CPO` (these use oracle representation automatically)

**Example command:**
```bash
python run.py \
  --env-id SafetyPointGoal1-v1 \
  --algo ShieldedTRPOLag \
  --prediction-horizon 1 \
  --penalty-type reward \
  --sampling-nbr 10 \
  --safety-bonus 1. \
  --use-wandb True \
  --fe-representation True \
  --use-acp True \
  --project-name shield 
```

#### Key Parameters:
- `--prediction-horizon`: 0 (no shielding), 1 (one-step ahead shielding)
- `--penalty-type`: `reward` (use SRO), `shield` (do not use SRO during optimization)
- `--fe-representation`: `True` (function encoder adaptation), `oracle` (ground-truth adaptation)
- `--sampling-nbr`: Number of action samples when adaptive shield is triggered
- `--safety-bonus`: Weight of safety in the augmented objective

#### Common Configurations:

| Mode                     | Parameters                                    |
|--------------------------|-----------------------------------------------|
| **SRO only**             | `--prediction-horizon 0 --penalty-type reward` |
| **Shield only**          | `--prediction-horizon 1 --penalty-type shield` |
| **Combined approach**    | `--prediction-horizon 1 --penalty-type reward` |

### 5. Run Unrolling Safety Layer (USL) Baseline

```bash
# Generic command
python run_usl.py --env <env_id> --use_usl --seed <seed> --oracle --save_model

# Example command
python run_usl.py --env SafetyPointGoal1-v1 --use_usl --seed 0 --oracle --save_model
```

### 6. Evaluate OOD Generalization

For OOD testing, use environments with level 2 (e.g., SafetyPointGoal2-v1). These environments have:
- 2 additional hazard spaces 
- Hidden parameters sampled from OOD range
- Use `prediction_horizon=1` to enable shielding, `0` to disable it

```bash
# Generic command
python 3.load_model.py <env_id> <algorithm> <seed> <sampling_nbr> <prediction_horizon>

# Example command
python 3.load_model.py SafetyPointGoal2-v1 ShieldedTRPO 0 100 1
```

## 🔍 Evaluation and Visualization

### Data Management

1. **Download data from Wandb:**
   ```bash
   # Use your project name (default: shield)
   python plot_wandb_data_download.py shield
   ```

2. **Organize downloaded data:**
   ```bash
   # For run.py experiments:
   python plot_organize_dir.py omni
   
   # For USL experiments:
   python plot_organize_dir.py skit
   ```

### Generate Analysis Plots

- **Research Questions:**
  ```bash
  python plot_rq1.py  # Performance comparison
  python plot_rq2.py  # OOD analysis
  python plot_rq3.py  # Core component evaluations
  ```

- **Ablation Studies:**
  ```bash
  python plot_abl_ppo.py
  python plot_abl_lag_lr.py
  python plot_abl_rep.py
  python plot_abl_training_hp.py
  python plot_abl_ood_hp.py
  ```

## 🔧 Advanced Configuration

### Function Encoder Settings

The Function Encoder's configurations can be modified in `configuration.py`, including:
- Network architecture
- Training parameters
- Inference settings

### Environment Parameters

For environment-specific hidden parameters and difficulty levels:
- Level 1 environments (e.g., SafetyPointGoal1-v1): Standard training environments
- Level 2 environments (e.g., SafetyPointGoal2-v1): OOD test environments with different parameter distributions

## ⚙️ Supported Algorithms

- **Shielding** (`omnisafe/algorithms/on_policy/shield/`):  
  - `ShieldedTRPOLag`  
  - `ShieldedPPOLag`  

- **Function Encoder Representations**:  
  - `CPO`, `PPOLag`, `TRPOLag`

- **Oracle Representations**:  
  - Any on-policy algorithm from [OmniSafe](https://github.com/PKU-Alignment/omnisafe)

## 📚 Acknowledgements

This code leverages and extends:
- [saferl_kit](https://github.com/zlr20/saferl_kit)
- [OmniSafe](https://github.com/PKU-Alignment/omnisafe)
- [Safe-Gym](https://github.com/PKU-Alignment/safety-gymnasium)

## 📜 License

Distributed under the MIT License. See `LICENSE` for details. 