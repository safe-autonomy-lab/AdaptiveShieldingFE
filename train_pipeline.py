#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from typing import List


def _run(cmd: List[str], env: dict | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def _load_env_for_eval() -> dict:
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        ld_lib = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib{(':' + ld_lib) if ld_lib else ''}"
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="SafetyHalfCheetahVelocity-v1")
    parser.add_argument("--algo", default="ShieldedRCPO")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--use-trained-policy", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--n-basis", type=int, default=4)
    parser.add_argument("--dynamics-model", default="fe")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--total-steps", type=int, default=int(2e6))
    parser.add_argument("--penalty-type", default="reward")
    parser.add_argument("--safety-bonus", type=float, default=1.0)
    parser.add_argument("--num-eval-episodes", type=int, default=100)
    parser.add_argument("--sampling-nbrs", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.275, 0.3])
    parser.add_argument("--idle-conditions", type=int, nargs="+", default=[4])
    parser.add_argument("--scales", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument("--skip-collect-transition", action="store_true", default=False)
    parser.add_argument("--skip-train-dynamics-predictor", action="store_true", default=False)
    parser.add_argument("--skip-run", action="store_true", default=False)
    parser.add_argument("--aggregate-root", default="ood_evaluation_folder")
    parser.add_argument("--aggregate-output", default="final_ood_results.json")
    args = parser.parse_args()

    env_info = args.env_id.split("-")[0]
    is_shielded_algo = args.algo.startswith("Shielded")
    if not is_shielded_algo:
        if not args.skip_collect_transition:
            print("Non-shielded algorithm detected; skipping transition collection.")
        if not args.skip_train_dynamics_predictor:
            print("Non-shielded algorithm detected; skipping dynamics predictor training.")
        args.skip_collect_transition = True
        args.skip_train_dynamics_predictor = True

    if not args.skip_collect_transition:
        _run(
            [
                sys.executable,
                "1.collect_transition.py",
                args.env_id,
                str(args.episodes),
                str(args.use_trained_policy),
                str(args.horizon),
            ],
        )

    if not args.skip_train_dynamics_predictor:
        _run(
            [
                sys.executable,
                "2.train_dynamics_predictor.py",
                "--env_id",
                args.env_id,
                "--dynamics_model",
                args.dynamics_model,
                "--n_basis",
                str(args.n_basis),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--prediction_horizon",
                str(args.horizon),
                "--seed",
                str(args.seed),
            ],
        )
        time.sleep(60)
        
    if not args.skip_run:
        _run(
            [
                sys.executable,
                "run.py",
                "--env-id",
                args.env_id,
                "--prediction-horizon",
                str(args.horizon),
                "--penalty-type",
                args.penalty_type,
                "--total-steps",
                str(args.total_steps),
                "--algo",
                args.algo,
                "--seed",
                str(args.seed),
                "--n-basis",
                str(args.n_basis),
                "--safety-bonus",
                str(args.safety_bonus),
                "--use-wandb",
                "True",
                "--project-name",
                "[clean]adaptive_shield",
            ],
        )
        time.sleep(60)
    
    _run(
        [
            sys.executable,
            "organize_run_folders.py",
            "--env-id",
            args.env_id,
            "--algorithm",
            args.algo,
            "--seed",
            str(args.seed),
        ],
    )

    _run(
        [
            sys.executable,
            "pareto_report.py",
            "--train-root",
            "results",
            "--train-output",
            "final_train_results.json",
        ],
    )

    eval_env = _load_env_for_eval()
    for sampling_nbr in args.sampling_nbrs:
        for threshold in args.thresholds:
            for idle_condition in args.idle_conditions:
                for scale in args.scales:
                    _run(
                        [
                            sys.executable,
                            "3.load_model.py",
                            args.env_id,
                            args.algo,
                            str(args.seed),
                            str(sampling_nbr),
                            str(args.horizon),
                            str(threshold),
                            str(idle_condition),
                            str(scale),
                            str(args.num_eval_episodes),
                            str(args.n_basis),
                        ],
                        env=eval_env,
                    )

    _run(
        [
            sys.executable,
            "pareto_report.py",
            "--root",
            f"ood_evaluation_folder/{env_info}",
            "--output",
            "pareto_report.csv",
        ],
    )

    if args.aggregate_root:
        time.sleep(5)
        _run(
            [
                sys.executable,
                "pareto_report.py",
                "--aggregate-root",
                args.aggregate_root,
                "--aggregate-output",
                args.aggregate_output,
            ],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
