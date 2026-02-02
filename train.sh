#!/usr/bin/env bash
set -euo pipefail

ENV_ID="SafetyHalfCheetahVelocity-v1"
# ENV_ID="SafetyCarCircle1-v1"
# ENV_ID="SafetyPointCircle1-v1"
ENV_INFO="${ENV_ID%-*}"
# ALGO_NAME="ShieldedTRPOLag"
ALGO_NAME="FOCOPS"
EPISODES=5
USE_TRAINED_POLICY=0
HORIZON=7
SEED=100
N_BASIS=16
# This decides; reward: shielding + sro, shield: only shielding, sro: only sro
# PENALTY_TYPE="reward" 
PENALTY_TYPE="reward"
# PENALTY_TYPE="sro"
# python run.py --env-id SafetyPointGoal1-v1 --prediction-horizon 7 --penalty-type reward --total-steps 10000 --algo ShieldedRCPO --seed 100 --n-basis 16
# python run.py --env-id SafetyHalfCheetahVelocity-v1 --prediction-horizon 7 --penalty-type reward --total-steps 10000 --algo ShieldedRCPO --seed 100 --n-basis 16
if [[ "$ALGO_NAME" == Shielded* ]]; then
  python 1.collect_transition.py "$ENV_ID" "$EPISODES" "$USE_TRAINED_POLICY" "$HORIZON"
else
  echo "Non-shielded algorithm detected; skipping transition collection."
fi

MODEL="fe"            # fe | transformer | pem | oracle
EPOCHS=2
BATCH_SIZE=8

if [[ "$ALGO_NAME" == Shielded* ]]; then
  python 2.train_dynamics_predictor.py \
  --env_id "$ENV_ID" \
  --dynamics_model "$MODEL" \
  --n_basis "$N_BASIS" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --prediction_horizon "$HORIZON" \
  --seed "$SEED"
else
  echo "Non-shielded algorithm detected; skipping dynamics predictor training."
fi

TOTAL_STEPS=20000

python run.py --env-id "$ENV_ID" --prediction-horizon "$HORIZON" --penalty-type "$PENALTY_TYPE" --total-steps "$TOTAL_STEPS" --algo "$ALGO_NAME" --seed "$SEED" --n-basis "$N_BASIS"

python organize_run_folders.py --env-id "$ENV_ID" --algorithm "$ALGO_NAME" --seed "$SEED"

python pareto_report.py --train-root "results" --train-output "final_train_results.json"

NUM_EVAL_EPISODES=10

# Grid search values (edit as needed)
SAMPLING_NBRS=(10)
THRESHOLDS=(0.25)
IDLE_CONDITIONS=(4)
SCALES=(0.05)

for SAMPLING_NBR in "${SAMPLING_NBRS[@]}"; do
  for THRESHOLD in "${THRESHOLDS[@]}"; do
    for IDLE_CONDITION in "${IDLE_CONDITIONS[@]}"; do
      for SCALE in "${SCALES[@]}"; do
        LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" python 3.load_model.py \
          "$ENV_ID" "$ALGO_NAME" "$SEED" \
          "$SAMPLING_NBR" "$HORIZON" "$THRESHOLD" "$IDLE_CONDITION" \
          "$SCALE" "$NUM_EVAL_EPISODES" "$N_BASIS"
      done
    done
  done
done

python pareto_report.py --root "ood_evaluation_folder/$ENV_INFO" --output "pareto_report.csv"
python pareto_report.py --aggregate-root "ood_evaluation_folder" --aggregate-output "final_ood_results.json"
