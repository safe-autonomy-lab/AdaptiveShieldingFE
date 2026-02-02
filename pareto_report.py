#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import shutil
import statistics
from typing import Dict, List, Tuple


def _read_metrics(csv_path: str) -> Dict[str, float]:
    metrics = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row.get("Metric")
            value = row.get("Value")
            if metric is None or value is None:
                continue
            try:
                metrics[metric] = float(value)
            except ValueError:
                continue
    return metrics


def _extract_params(folder_name: str) -> Dict[str, str]:
    # Expected pattern: Shield_s10_th0.25_idle4_h1_scale0.03
    parts = folder_name.split("_")
    params = {}
    for part in parts:
        if part.startswith("s"):
            params["sampling_nbr"] = part[1:]
        elif part.startswith("th"):
            params["threshold"] = part[2:]
        elif part.startswith("idle"):
            params["idle_condition"] = part[4:]
        elif part.startswith("h"):
            params["prediction_horizon"] = part[1:]
        elif part.startswith("scale"):
            params["scale"] = part[5:]
    return params


def _pareto_front(points: List[Tuple[float, float]]) -> List[bool]:
    # Each point: (cost, reward). Pareto: lower cost, higher reward.
    n = len(points)
    is_pareto = [True] * n
    for i, (cost_i, reward_i) in enumerate(points):
        for j, (cost_j, reward_j) in enumerate(points):
            if i == j:
                continue
            if cost_j <= cost_i and reward_j >= reward_i and (cost_j < cost_i or reward_j > reward_i):
                is_pareto[i] = False
                break
    return is_pareto


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def _load_env_pareto_report(csv_path: str) -> Dict[str, List[Dict[str, float]]]:
    grouped: Dict[str, List[Dict[str, float]]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reward = _safe_float(row.get("reward"))
            cost = _safe_float(row.get("cost"))
            if not (reward == reward and cost == cost):
                continue
            folder = row.get("folder") or ""
            seed_dir = row.get("seed_dir") or ""
            if not folder and seed_dir:
                folder = os.path.dirname(seed_dir)
            if not folder:
                continue
            grouped.setdefault(folder, []).append({"reward": reward, "cost": cost})
    return grouped


def _summarize_group(folder: str, entries: List[Dict[str, float]]) -> Dict[str, object]:
    rewards = [e["reward"] for e in entries]
    costs = [e["cost"] for e in entries]
    config_name = os.path.basename(folder)
    params = _extract_params(config_name)
    return {
        "config": config_name,
        "folder": folder,
        "n_seeds": len(entries),
        "reward": {"mean": statistics.mean(rewards), "std": _std(rewards)},
        "cost": {"mean": statistics.mean(costs), "std": _std(costs)},
        "params": params,
    }


def _algo_key_from_config(config_name: str) -> str:
    if "_s" in config_name:
        return config_name.split("_s", 1)[0]
    return config_name


def _aggregate_root(root: str, output: str) -> str:
    results: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for name in sorted(os.listdir(root)):
        env_dir = os.path.join(root, name)
        if not os.path.isdir(env_dir):
            continue
        csv_path = os.path.join(env_dir, "pareto_report.csv")
        if not os.path.isfile(csv_path):
            continue
        grouped = _load_env_pareto_report(csv_path)
        summaries = [_summarize_group(folder, entries) for folder, entries in grouped.items()]
        if not summaries:
            results[name] = {}
            continue
        algo_groups: Dict[str, List[Dict[str, object]]] = {}
        for summary in summaries:
            algo_key = _algo_key_from_config(summary["config"])
            algo_groups.setdefault(algo_key, []).append(summary)
        env_result: Dict[str, List[Dict[str, object]]] = {}
        for algo_key, algo_summaries in algo_groups.items():
            points = [(s["cost"]["mean"], s["reward"]["mean"]) for s in algo_summaries]
            pareto_flags = _pareto_front(points)
            pareto = [s for s, flag in zip(algo_summaries, pareto_flags) if flag]
            pareto_sorted = sorted(pareto, key=lambda x: (x["cost"]["mean"], -x["reward"]["mean"]))
            env_result[algo_key] = pareto_sorted
        results[name] = env_result

    out_path = os.path.join(root, output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    return out_path


def _read_progress_metrics(csv_path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep_ret = _safe_float(row.get("Metrics/EpRet"))
            ep_cost = _safe_float(row.get("Metrics/EpCost"))
            epoch = _safe_float(row.get("Train/Epoch"))
            if not (ep_ret == ep_ret and ep_cost == ep_cost and epoch == epoch):
                continue
            rows.append({"epoch": epoch, "reward": ep_ret, "cost": ep_cost})
    return rows


def _seed_last_k_mean(rows: List[Dict[str, float]], k: int = 10) -> Dict[str, float] | None:
    if not rows:
        return None
    rows_sorted = sorted(rows, key=lambda r: r["epoch"])
    tail = rows_sorted[-k:]
    rewards = [r["reward"] for r in tail]
    costs = [r["cost"] for r in tail]
    return {"reward": statistics.mean(rewards), "cost": statistics.mean(costs)}


def _aggregate_training(root: str, output: str, k_last: int = 10) -> str:
    results: Dict[str, List[Dict[str, object]]] = {}
    for model_root in sorted(os.listdir(root)):
        model_dir = os.path.join(root, model_root)
        if not os.path.isdir(model_dir):
            continue
        for env_name in sorted(os.listdir(model_dir)):
            env_dir = os.path.join(model_dir, env_name)
            if not os.path.isdir(env_dir):
                continue
            for horizon_name in sorted(os.listdir(env_dir)):
                horizon_dir = os.path.join(env_dir, horizon_name)
                if not os.path.isdir(horizon_dir) or not horizon_name.startswith("h"):
                    continue
                for algo_name in sorted(os.listdir(horizon_dir)):
                    algo_dir = os.path.join(horizon_dir, algo_name)
                    if not os.path.isdir(algo_dir):
                        continue
                    seed_dirs = glob.glob(os.path.join(algo_dir, "seed*"))
                    seed_means: List[Dict[str, float]] = []
                    for seed_dir in seed_dirs:
                        progress_path = os.path.join(seed_dir, "progress.csv")
                        if not os.path.isfile(progress_path):
                            continue
                        rows = _read_progress_metrics(progress_path)
                        mean_vals = _seed_last_k_mean(rows, k=k_last)
                        if mean_vals is None:
                            continue
                        seed_means.append(mean_vals)
                    if not seed_means:
                        continue
                    rewards = [s["reward"] for s in seed_means]
                    costs = [s["cost"] for s in seed_means]
                    entry = {
                        "algorithm": algo_name,
                        "folder": algo_dir,
                        "n_seeds": len(seed_means),
                        "reward": {"mean": statistics.mean(rewards), "std": _std(rewards)},
                        "cost": {"mean": statistics.mean(costs), "std": _std(costs)},
                        "horizon": horizon_name,
                        "model_root": model_root,
                    }
                    results.setdefault(env_name, []).append(entry)

    pareto_results: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for env_name, entries in results.items():
        if not entries:
            pareto_results[env_name] = {}
            continue
        algo_groups: Dict[str, List[Dict[str, object]]] = {}
        for entry in entries:
            algo_groups.setdefault(entry["algorithm"], []).append(entry)
        env_result: Dict[str, List[Dict[str, object]]] = {}
        for algo_name, algo_entries in algo_groups.items():
            points = [(e["cost"]["mean"], e["reward"]["mean"]) for e in algo_entries]
            pareto_flags = _pareto_front(points)
            pareto = [e for e, flag in zip(algo_entries, pareto_flags) if flag]
            pareto_sorted = sorted(pareto, key=lambda x: (x["cost"]["mean"], -x["reward"]["mean"]))
            env_result[algo_name] = pareto_sorted
        pareto_results[env_name] = env_result

    out_path = os.path.join(root, output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pareto_results, f, indent=2, sort_keys=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="ood_evaluation_folder/<ENV_NAME>")
    parser.add_argument("--metric-reward", default="Average episode reward")
    parser.add_argument("--metric-cost", default="Average episode cost")
    parser.add_argument("--output", default="pareto_report.csv")
    parser.add_argument("--best-root", default="", help="best_ood_evaluation_folder/<ENV_NAME>")
    parser.add_argument("--aggregate-root", default="", help="ood_evaluation_folder (aggregate env-level pareto_report.csv)")
    parser.add_argument("--aggregate-output", default="final_ood_results.json")
    parser.add_argument("--train-root", default="", help="results (aggregate training progress.csv)")
    parser.add_argument("--train-output", default="final_train_results.json")
    parser.add_argument("--train-last-k", type=int, default=10)
    args = parser.parse_args()

    if args.aggregate_root:
        out_path = _aggregate_root(args.aggregate_root, args.aggregate_output)
        print(f"Saved {out_path}")
        return 0
    if args.train_root:
        out_path = _aggregate_training(args.train_root, args.train_output, k_last=args.train_last_k)
        print(f"Saved {out_path}")
        return 0
    if not args.root:
        parser.error("--root is required unless --aggregate-root is provided")

    candidates = []
    for folder in sorted(glob.glob(os.path.join(args.root, "*"))):
        if not os.path.isdir(folder):
            continue
        seed_dirs = glob.glob(os.path.join(folder, "seed*"))
        for seed_dir in seed_dirs:
            csv_path = os.path.join(seed_dir, "evaluation_results.csv")
            if not os.path.isfile(csv_path):
                continue
            metrics = _read_metrics(csv_path)
            if args.metric_reward not in metrics or args.metric_cost not in metrics:
                continue
            folder_name = os.path.basename(folder)
            params = _extract_params(folder_name)
            candidates.append({
                "folder": folder,
                "seed_dir": seed_dir,
                "reward": metrics[args.metric_reward],
                "cost": metrics[args.metric_cost],
                **params,
            })

    if not candidates:
        print("No evaluation_results.csv found.")
        return 1

    points = [(c["cost"], c["reward"]) for c in candidates]
    pareto_flags = _pareto_front(points)
    for c, flag in zip(candidates, pareto_flags):
        c["pareto"] = flag

    out_path = os.path.join(args.root, args.output)
    fieldnames = ["folder", "seed_dir", "reward", "cost", "pareto", "sampling_nbr", "threshold", "idle_condition", "prediction_horizon", "scale"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:
            writer.writerow({k: c.get(k, "") for k in fieldnames})

    pareto = [c for c in candidates if c["pareto"]]
    pareto_sorted = sorted(pareto, key=lambda x: (x["cost"], -x["reward"]))
    print("Pareto front (cost asc, reward desc):")
    for c in pareto_sorted:
        print(f"- cost={c['cost']:.4f} reward={c['reward']:.4f} folder={os.path.basename(c['folder'])} seed={os.path.basename(c['seed_dir'])}")
    print(f"Saved report to {out_path}")

    if args.best_root:
        os.makedirs(args.best_root, exist_ok=True)
        with open(out_path, newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row.get("pareto") not in ("True", "true", "1"):
                continue
            folder = row.get("folder", "")
            seed_dir = row.get("seed_dir", "")
            if not folder or not seed_dir:
                continue
            dst = os.path.join(args.best_root, os.path.basename(folder), os.path.basename(seed_dir))
            os.makedirs(dst, exist_ok=True)
            for name in ("evaluation_results.csv", "episode_data.npz"):
                src = os.path.join(seed_dir, name)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(dst, name))
        print(f"Best configs copied to {args.best_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
