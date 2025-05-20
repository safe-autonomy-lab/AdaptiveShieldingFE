"""Script to generate reward vs. cost trade-off scatter plots comparing algorithm hyperparameters."""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from typing import Dict, List, Optional, Tuple

# --- Data Loading and Processing Functions (Adapted from previous script) ---

def find_csv_files(
    base_folder: str,
    robot_type: str,
    env_type: str,
    level: str,
) -> List[str]:
    """Find all evaluation_results.csv files for a specific environment setup."""
    csv_files = []
    env_folder = f"Safety{robot_type}{env_type}{level}"
    env_path = os.path.join(base_folder, env_folder)

    if not os.path.isdir(env_path):
        print(f"Warning: Environment folder not found: {env_path}")
        return []

    # Find all algorithm folders within the environment folder
    for algo_folder in os.listdir(env_path):
        algo_path = os.path.join(env_path, algo_folder)
        if os.path.isdir(algo_path):
            for seed_folder in os.listdir(algo_path):
                if seed_folder.startswith("seed"):
                    seed_path = os.path.join(algo_path, seed_folder)
                    if os.path.isdir(seed_path):
                        result_file = os.path.join(seed_path, "evaluation_results.csv")
                        if os.path.isfile(result_file):
                            csv_files.append(result_file)
    return csv_files

def process_results(csv_files: List[str]) -> pd.DataFrame:
    """Read CSV files, parse info, and aggregate results."""
    all_data = []
    seed_pattern = re.compile(r"^seed(\d+)$")

    for file_path in csv_files:
        try:
            parts = file_path.split(os.sep)
            if len(parts) >= 5:
                env_name = parts[-4]
                algorithm_name = parts[-3]
                seed_folder = parts[-2]
                match = seed_pattern.match(seed_folder)
                if not match:
                    print(f"Warning: Could not parse seed folder: {seed_folder}")
                    continue
                seed = int(match.group(1))

                df = pd.read_csv(file_path)
                df["environment"] = env_name
                df["algorithm"] = algorithm_name
                df["seed"] = seed
                all_data.append(df)
            else:
                print(f"Warning: Unexpected file path structure: {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    grouped = combined_df.groupby(["environment", "algorithm", "Metric"])["Value"]
    summary_df = grouped.agg(["mean", "std"]).reset_index()
    return summary_df

# --- New Naming and Styling Logic ---

def simplify_hyperparameter_name(algo_name: str) -> Tuple[str, str]:
    """Simplify algorithm name to identify base type and hyperparameter.
    
    Returns:
        Tuple[str, str]: (base_algorithm_type, hyperparameter_label)
                       e.g., ("PPO", "PPO+Shield (N=5)")
    """
    suffix_pattern = re.compile(r"^(.*?)_(\d+)_(\d+)$")
    match = suffix_pattern.match(algo_name)

    base_name_part = algo_name
    hyperparam_label = ""
    sampling_num = -1
    cost_coeff_num = -1 # Initialize to distinguish from 0

    if match:
        base_name_part = match.group(1)
        sampling_num = int(match.group(2))
        cost_coeff_num = int(match.group(3))
        # Removed cost_coeff_zero logic from here, handle per-algo type

    # Determine Base Algorithm and Final Label
    base_algo_type_inferred = "TRPO" if "TRPO" in base_name_part else "PPO"

    simple_base = base_name_part # Default

    # Prioritize Base Lag identification
    if base_name_part == "PPOLag" or base_name_part == "TRPOLag":
        simple_base = base_name_part
        hyperparam_label = "(Base Lag)" # Assign Base Lag label directly
    elif "ACP" in base_name_part and "PPO" in base_name_part:
        simple_base = "Shield"
        hyperparam_label = f"(N={sampling_num})" # Assumes suffix exists
    elif "ACP" in base_name_part and "TRPO" in base_name_part:
        simple_base = "Shield"
        hyperparam_label = f"(N={sampling_num})" # Assumes suffix exists
    elif "ShieldedPPO" in base_name_part:
        simple_base = "SRO+Shield"
        hyperparam_label = f"(N={sampling_num})" # Assumes suffix exists
    elif "ShieldedTRPO" in base_name_part:
        simple_base = "SRO+Shield"
        hyperparam_label = f"(N={sampling_num})" # Assumes suffix exists
    elif "SafetyObjOnly_PPO" in base_name_part:
        simple_base = "Safe"
        # Handle specific suffix for SafetyObjOnly
        if sampling_num == 10 and cost_coeff_num == 0:
             hyperparam_label = "(N=10, C=0)"
        elif sampling_num != -1: # If suffix exists but not _10_0
             hyperparam_label = f"(N={sampling_num})"
        else: # No suffix?
             hyperparam_label = "(Unknown Hyperparam)"
    elif "SafetyObjOnly_TRPO" in base_name_part:
        simple_base = "Safe"
        # Handle specific suffix for SafetyObjOnly
        if sampling_num == 10 and cost_coeff_num == 0:
             hyperparam_label = "(N=10, C=0)"
        elif sampling_num != -1:
             hyperparam_label = f"(N={sampling_num})"
        else:
             hyperparam_label = "(Unknown Hyperparam)"
    else:
        # Fallback for other cases, potentially with suffixes
        if sampling_num != -1:
             if sampling_num == 10 and cost_coeff_num == 0:
                 hyperparam_label = f"(N={sampling_num}, C=0)"
             else:
                 hyperparam_label = f"(N={sampling_num})"
        # Check again if it's Lag without suffix (shouldn't happen with current data)
        elif algo_name.endswith("Lag"):
             hyperparam_label = "(Base Lag)"
        else: # True fallback
             hyperparam_label = "(Unknown Hyperparam)"


    final_label = f"{simple_base} {hyperparam_label}"

    return base_algo_type_inferred, final_label # Return inferred base_algo_type

def get_hyperparameter_style(hyperparam_label: str) -> Dict[str, any]:
    """Get color and marker based on the hyperparameter label."""
    # Define styles for each hyperparameter configuration
    styles = {
        # N=5
        "(N=5)": {"color": to_rgba("#1f77b4"), "marker": "o"},  # Blue circle
        # N=10
        "(N=10)": {"color": to_rgba("#ff7f0e"), "marker": "X"},  # Orange X
        # N=50
        "(N=50)": {"color": to_rgba("#2ca02c"), "marker": "^"},  # Green triangle_up
        # N=100
        "(N=100)": {"color": to_rgba("#9467bd"), "marker": "D"},  # Purple diamond
        # N=10, C=0 (Only for specific algos like SafetyObjOnly now)
        # This key might still be generated by simplify_hyperparameter_name for non-Lag algos
        "(N=10, C=0)": {"color": to_rgba("#d62728"), "marker": "s"}, # Red square
        # Updated style for Base Lag as requested
        "(Base Lag)": {"color": to_rgba("#d62728"), "marker": "s"}, # Red square
        "(Unknown Hyperparam)": {"color": to_rgba("#7f7f7f"), "marker": "*"} # Gray star
    }

    # Find the relevant part of the label (e.g., "(N=5)")
    match = re.search(r"\(.*\)", hyperparam_label)
    key = match.group(0) if match else "(Unknown Hyperparam)"

    return styles.get(key, styles["(Unknown Hyperparam)"])

# --- New Plotting Function ---

def plot_hyperparameter_comparison(
    standard_folder: str,
    base_algo_type: str, # "PPO" or "TRPO"
    output_file: Optional[str] = None,
) -> None:
    """
    Create a 2x4 comparison plot of hyperparameters for a base algorithm type 
    at Level 2, comparing Car and Point robots across different environments.

    Args:
        standard_folder: Folder containing standard evaluation data.
        base_algo_type: The base algorithm family to focus on ("PPO" or "TRPO").
        output_file: Path to save the output image.

    Layout:
        Rows: Car, Point
        Columns: Goal, Button, Push, Circle
        Level: 2 (fixed)
    """
    # sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    try:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })
    except:
        print("LaTeX rendering not available, using default text rendering")

    FIXED_LEVEL: str = "2"
    PLOT_ROBOT_TYPES: List[str] = ["Car", "Point"]
    PLOT_ENV_TYPES: List[str] = ["Goal", "Button", "Push", "Circle"]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=False, sharey=False) # Adjusted figsize
    all_hyperparam_labels = set() # Store unique labels for the legend

    for row_idx, current_robot_type in enumerate(PLOT_ROBOT_TYPES):
        for col_idx, current_env_type in enumerate(PLOT_ENV_TYPES):
            ax = axes[row_idx, col_idx]
            csv_files = find_csv_files(standard_folder, current_robot_type, current_env_type, FIXED_LEVEL)

            if not csv_files:
                print(f"No CSV files found for {current_robot_type}{current_env_type}{FIXED_LEVEL}")
                ax.text(0.5, 0.5, f"No data files found\\n{current_robot_type} {current_env_type}", 
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_xticks([]) # Hide ticks if no data
                ax.set_yticks([])
                # Set titles and labels even for empty plots for consistency
                ax.set_title(f"\\textbf{{{current_robot_type} - {current_env_type}}}", fontsize=18)
                if col_idx == 0: ax.set_ylabel("Return", fontsize=12)
                if row_idx == (len(PLOT_ROBOT_TYPES) - 1): ax.set_xlabel("Cost Rate (\%)", fontsize=12) 
                continue

            summary_df = process_results(csv_files)
            if summary_df.empty:
                print(f"No results processed for {current_robot_type}{current_env_type}{FIXED_LEVEL}")
                ax.text(0.5, 0.5, f"No results found\\n{current_robot_type} {current_env_type}", 
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"\\textbf{{{current_robot_type} - {current_env_type}}}", fontsize=18)
                if col_idx == 0: ax.set_ylabel("Return", fontsize=12)
                if row_idx == (len(PLOT_ROBOT_TYPES) - 1): ax.set_xlabel("Cost Rate (\%)", fontsize=12)
                continue

            # --- Filter for the specific algorithm types --- 
            allowed_algos = set()
            if base_algo_type == "PPO":
                allowed_algos = {
                    "PPOLag_10_0",
                    "ShieldedPPO_5_1",
                    "ShieldedPPO_10_1",
                    "ShieldedPPO_50_1",
                    "ShieldedPPO_100_1",
                }
            elif base_algo_type == "TRPO":
                 allowed_algos = {
                    "TRPOLag_10_0",
                    "ShieldedTRPO_5_1",
                    "ShieldedTRPO_10_1",
                    "ShieldedTRPO_50_1",
                    "ShieldedTRPO_100_1",
                }
            else:
                print(f"Warning: Unknown base_algo_type: {base_algo_type}. Skipping subplot.")
                ax.text(0.5, 0.5, f"Unknown Algorithm Type\\n{base_algo_type}",
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"\\textbf{{{current_robot_type} - {current_env_type}}}", fontsize=14)
                if col_idx == 0: ax.set_ylabel("Return", fontsize=12)
                if row_idx == (len(PLOT_ROBOT_TYPES) - 1): ax.set_xlabel("Cost Rate (\%)", fontsize=12)
                continue

            filtered_df = summary_df[summary_df['algorithm'].isin(allowed_algos)].copy()
            
            if filtered_df.empty:
                print(f"No target algorithms ({', '.join(sorted(list(allowed_algos)))}) found for {current_robot_type}{current_env_type}{FIXED_LEVEL}")
                ax.text(0.5, 0.5, f"No target {base_algo_type} data\\n{current_robot_type} {current_env_type}", 
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"\\textbf{{{current_robot_type} - {current_env_type}}}", fontsize=14)
                if col_idx == 0: ax.set_ylabel("Return", fontsize=12)
                if row_idx == (len(PLOT_ROBOT_TYPES) - 1): ax.set_xlabel("Cost Rate (\%)", fontsize=12)
                continue

            # --- Process and Plot Filtered Data ---            
            reward_cost_data = []
            algorithms_in_subplot = filtered_df["algorithm"].unique()

            for algo in algorithms_in_subplot:
                # Use the passed base_algo_type for consistency, simplify_hyperparameter_name returns inferred one
                _, display_name = simplify_hyperparameter_name(algo)
                algo_data = filtered_df[filtered_df["algorithm"] == algo]
                
                reward_data = algo_data[algo_data["Metric"] == "Average episode reward"]
                cost_data = algo_data[algo_data["Metric"] == "Average episode cost"]

                if not reward_data.empty and not cost_data.empty:
                    reward_mean = reward_data["mean"].values[0]
                    reward_std = reward_data["std"].values[0]
                    cost_mean = cost_data["mean"].values[0]
                    cost_std = cost_data["std"].values[0]
                    cost_rate = cost_mean / 500 * 100 if "Circle" in current_env_type else cost_mean / 1000 * 100
                    
                    reward_cost_data.append({
                        "algorithm": algo,
                        "display_name": display_name,
                        "reward_mean": reward_mean,
                        "reward_std": reward_std,
                        "cost_mean": cost_mean,
                        "cost_std": cost_std,
                        "cost_rate": cost_rate
                    })
                    all_hyperparam_labels.add(display_name)
                else:
                     print(f"Warning: Missing reward or cost for {algo} in {current_robot_type}{current_env_type}{FIXED_LEVEL}")

            # Plot scatter with error bars
            plotted_labels_in_subplot = set()
            if not reward_cost_data:
                 ax.text(0.5, 0.5, f"No plottable data\\n{current_robot_type} {current_env_type}", 
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
                 ax.set_xticks([])
                 ax.set_yticks([])
            else:
                for data_point in reward_cost_data:
                    display_name = data_point["display_name"]
                    style = get_hyperparameter_style(display_name)
                    label_for_legend = display_name if display_name not in plotted_labels_in_subplot else ""
                    plotted_labels_in_subplot.add(display_name)

                    ax.scatter(
                        data_point["cost_rate"], data_point["reward_mean"],
                        label=label_for_legend, # Only label once for the shared legend
                        color=style["color"],
                        marker=style["marker"],
                        s=100, alpha=1.0, edgecolors='black', linewidth=1
                    )
                    # Removed errorbar plotting - only plotting mean values

            # Configure subplot titles and labels
            ax.set_title(f"\\textbf{{{current_robot_type}-{current_env_type}}}", fontsize=18)
            if col_idx == 0: ax.set_ylabel("Return", fontsize=12)
            if row_idx == (len(PLOT_ROBOT_TYPES) - 1): ax.set_xlabel("Cost Rate (\%)", fontsize=12)
            
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set axis limits if data exists
            costs_in_subplot = [d["cost_rate"] for d in reward_cost_data]
            rewards_in_subplot = [d["reward_mean"] for d in reward_cost_data]
            if costs_in_subplot and rewards_in_subplot:
                min_cost, max_cost = min(costs_in_subplot), max(costs_in_subplot)
                min_reward, max_reward = min(rewards_in_subplot), max(rewards_in_subplot)
                
                cost_range = max_cost - min_cost
                reward_range = max_reward - min_reward

                cost_pad = cost_range * 0.1 if cost_range > 1e-6 else max(0.1, abs(min_cost)*0.1 if abs(min_cost) > 1e-6 else 0.1)
                reward_pad = reward_range * 0.1 if reward_range > 1e-6 else max(0.1, abs(max_reward)*0.1 if abs(max_reward) > 1e-6 else 0.1)
                
                ax.set_xlim(min_cost - cost_pad, max_cost + cost_pad)
                ax.set_ylim(min_reward - reward_pad, max_reward + reward_pad)
            # Removed ax.invert_xaxis() from else block

    # --- Create Unified Legend --- 
    if all_hyperparam_labels:
        sorted_labels = sorted(list(all_hyperparam_labels), key=lambda x: (re.search(r"N=(\d+)", x).group(1) if re.search(r"N=(\d+)", x) else 'ZZZ', x))
        
        legend_handles = []
        for label_text in sorted_labels:
            style = get_hyperparameter_style(label_text)
            handle = plt.Line2D(
                [0], [0], marker=style["marker"], color='w', 
                markerfacecolor=style["color"], markeredgecolor='black', markersize=10
            )
            legend_handles.append(handle)

        fig.legend(
            legend_handles, sorted_labels, 
            loc='upper center', bbox_to_anchor=(0.5, 0.07), # Adjusted for 2 rows, might need tuning
            ncol=min(len(sorted_labels), 5),
            frameon=True, fontsize=12
        )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95]) # Adjust rect to make space for legend and suptitle

    if output_file:
        os.makedirs("./plot_ablations", exist_ok=True)
        # Construct full path for saving if output_file is just a name
        save_path = os.path.join("./plot_ablations", output_file)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    # plt.show()
    plt.close(fig) # Close the figure to free memory

# --- Main Execution Logic ---

def main():
    """Main function to generate all hyperparameter comparison plots."""
    standard_folder = "./ood_evaluation_folder"
    base_algo_types = ["PPO", "TRPO"]
    
    for base_algo in base_algo_types:
        output_filename = f"eval_hp_comparison_{base_algo.lower()}_level2_car_point.png"
        print(f"\n--- Generating plot: {output_filename} ---")
        
        plot_hyperparameter_comparison(
            standard_folder=standard_folder,
            base_algo_type=base_algo,
            output_file=output_filename
        )

if __name__ == "__main__":
    main()