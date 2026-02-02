#!/usr/bin/env python3
import argparse
import glob
import json
import os
import shutil


def _find_latest_epoch_file(torch_save_dir: str) -> str | None:
    candidates = glob.glob(os.path.join(torch_save_dir, "epoch-*.pt"))
    best_epoch = -1
    best_path = None
    for path in candidates:
        name = os.path.basename(path)
        try:
            epoch = int(name.split("-")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        if epoch > best_epoch:
            best_epoch = epoch
            best_path = path
    return best_path


def _find_latest_run_dir(base_dir: str, seed: int) -> str | None:
    seed_str = str(seed).zfill(3)
    pattern = os.path.join(base_dir, f"seed-{seed_str}-*")
    run_dirs = sorted(glob.glob(pattern))
    if not run_dirs:
        return None
    return run_dirs[-1]


def _organize_run_artifacts(run_dir: str, env_id: str, algorithm: str, seed: int) -> str | None:
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path, "r") as f:
        cfg = json.load(f)

    shield_cfgs = cfg.get("shield_cfgs", {})
    use_fe = bool(shield_cfgs.get("use_fe_representation", False))
    dynamics_model = shield_cfgs.get("dynamics_model", "fe")
    prediction_horizon = int(shield_cfgs.get("prediction_horizon", 1))
    folder_name = dynamics_model if use_fe else "oracle"

    penalty_type = str(shield_cfgs.get("penalty_type", "")).lower()
    algo_name = algorithm
    if penalty_type == "reward":
        algo_name = f"{algorithm}withSRO"
    elif penalty_type == "sro":
        base_algo = algorithm
        if base_algo.startswith("Shielded"):
            base_algo = base_algo[len("Shielded") :]
        algo_name = f"{base_algo}withSRO"
    elif penalty_type == "shield":
        algo_name = algorithm

    results_dir = os.path.join("results", folder_name, env_id, f"h{prediction_horizon}", algo_name, f"seed{seed}")
    torch_save_src = os.path.join(run_dir, "torch_save")
    torch_save_dst = os.path.join(results_dir, "torch_save")
    os.makedirs(torch_save_dst, exist_ok=True)

    latest_epoch = _find_latest_epoch_file(torch_save_src)
    if latest_epoch is None:
        return None
    dst_epoch = os.path.join(torch_save_dst, os.path.basename(latest_epoch))
    if os.path.isfile(dst_epoch):
        src_mtime = os.path.getmtime(latest_epoch)
        dst_mtime = os.path.getmtime(dst_epoch)
        if src_mtime > dst_mtime:
            os.remove(dst_epoch)
            shutil.move(latest_epoch, dst_epoch)
    else:
        shutil.move(latest_epoch, dst_epoch)

    dst_config = os.path.join(results_dir, "config.json")
    if os.path.isfile(dst_config):
        if os.path.getmtime(config_path) > os.path.getmtime(dst_config):
            shutil.copy2(config_path, dst_config)
    else:
        shutil.copy2(config_path, dst_config)

    progress_path = os.path.join(run_dir, "progress.csv")
    dst_progress = os.path.join(results_dir, "progress.csv")
    if os.path.isfile(progress_path):
        if os.path.isfile(dst_progress):
            if os.path.getmtime(progress_path) > os.path.getmtime(dst_progress):
                shutil.copy2(progress_path, dst_progress)
        else:
            shutil.copy2(progress_path, dst_progress)

    return results_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    runs_bases = [
        f"./runs/{args.algorithm}-{args.env_id}",
        f"./runs/{args.algorithm}-{{{args.env_id}}}",
    ]
    run_dir = None
    for runs_base in runs_bases:
        run_dir = _find_latest_run_dir(runs_base, args.seed)
        if run_dir:
            break
    if run_dir is None:
        raise FileNotFoundError(f"No run dir found for seed {args.seed} under {runs_bases}")

    results_dir = _organize_run_artifacts(run_dir, args.env_id, args.algorithm, args.seed)
    if results_dir is None:
        raise FileNotFoundError(f"Missing config.json or epoch-*.pt in {run_dir}")
    print(f"Organized artifacts into {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
