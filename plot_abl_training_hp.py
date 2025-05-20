import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from typing import Dict, List, Tuple
from plot_configuration import FONT_SIZES


def calculate_moving_window_stats(csv_path: str, window_size: int = 100) -> pd.DataFrame:
    """
    Calculate moving window statistics from CSV log file.
    
    Args:
        csv_path: Path to the CSV file
        window_size: Size of the moving window
        
    Returns:
        DataFrame with moving window statistics
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Calculate moving window statistics
    stats = pd.DataFrame()
    stats['total_steps'] = df['total_steps']
    
    # Calculate moving averages for each metric
    for column in ['EpRet', 'EpCost']:
        stats[f'{column}_Mean'] = df[column].rolling(window=window_size, min_periods=1).mean()
        stats[f'{column}_Std'] = df[column].rolling(window=window_size, min_periods=1).std()
        stats[f'{column}_Min'] = df[column].rolling(window=window_size, min_periods=1).min()
        stats[f'{column}_Max'] = df[column].rolling(window=window_size, min_periods=1).max()
    
    return stats

def parse_experiment_path(path: str) -> Tuple[str, str, int]:
    """Parse experiment path to extract algorithm, environment, and seed."""
    path_str = str(path)
    # Update pattern to match Shield_s4_b1 format
    pattern = r".*/([A-Za-z]+[A-Za-z0-9_.-]+)/([A-Za-z]+_s\d+_b[\d.]+_t[\d.]+|[A-Za-z]+)/seed(\d+)"
    print(f"Trying to match pattern on path: {path_str}")
    match = re.search(pattern, path_str)
    if match:
        env = match.group(1)
        alg = match.group(2)
        seed = int(match.group(3))
        print(f"Successfully parsed: env={env}, alg={alg}, seed={seed}")
        return alg, env, seed
    
    raise ValueError(f"Invalid path format: {path}")

def load_and_process_experiments(base_path: str) -> Dict[str, Dict[str, List[pd.DataFrame]]]:
    """Load and organize experiment data."""
    base_dir = Path(base_path)
    experiments: Dict[str, Dict[str, List[pd.DataFrame]]] = {}
    
    # Add debug print to see what we're looking for
    print(f"\nSearching in base directory: {base_dir.absolute()}")
    
    # Find all logger.csv files recursively
    for log_name in ["logger.csv", "progress.csv", "history.csv"]:
        print(f"\nLooking for {log_name} files...")
        
        for exp_dir in base_dir.rglob(log_name):
            try:
                # Print the full path being processed
                print(f"\nProcessing file: {exp_dir}")
                print(f"Parent directory: {exp_dir.parent}")
                
                alg, env, seed = parse_experiment_path(str(exp_dir.parent))
                print(f"Parsed: env={env}, alg={alg}, seed={seed}")
                
                # Initialize nested dictionary structure
                if env not in experiments:
                    experiments[env] = {}
                if alg not in experiments[env]:
                    experiments[env][alg] = []
                
                # Load and store DataFrame
                df = pd.read_csv(str(exp_dir))
                print(f"Loaded CSV with shape: {df.shape}")
                print(f"Columns: {df.columns.tolist()}")
                
                processed_df = pd.DataFrame()
                if 'EpRet' in df.columns and 'EpCost' in df.columns:
                    print("Using logger.csv format")
                    processed_df = calculate_moving_window_stats(str(exp_dir))
                elif 'Metrics/EpRet' in df.columns and 'Metrics/EpCost' in df.columns:
                    print("Using progress.csv format")
                    processed_df['EpRet_Mean'] = df['Metrics/EpRet']
                    processed_df['EpCost_Mean'] = df['Metrics/EpCost']
                else:
                    print(f"Warning: Required columns not found. Available columns: {df.columns.tolist()}")
                    continue

                if not processed_df.empty:
                    experiments[env][alg].append(processed_df)
                    print(f"Successfully added data for {env}/{alg}/seed{seed}")
                else:
                    print("Warning: Processed DataFrame is empty")
                
            except Exception as e:
                print(f"Error processing {exp_dir}: {e}")
                import traceback
                print(traceback.format_exc())
    
    # Print summary of loaded data
    print("\nLoaded data summary:")
    for env in experiments:
        print(f"\nEnvironment: {env}")
        for alg in experiments[env]:
            print(f"  Algorithm: {alg} - {len(experiments[env][alg])} seeds")
            
    return experiments

def create_ablation_study_plot_shield(
    experiments: Dict[str, Dict[str, List[pd.DataFrame]]],
    robot_type: str,
    study_type: str,  # "sampling" or "bonus"
    fixed_value: float,
    metrics: List[str] = ["EpRet", "EpCost"],
    every_n_episode: int = 20,
    exclude_envs: List[str] = ["Doggo"]
) -> plt.Figure:
    """Create a 2x4 grid of plots for ablation studies of shielding parameters."""
    # Set up LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
    # sns.set_style("whitegrid")
    
    # Define configurations and display names for different ablation studies
    if study_type == "sampling":
        fixed_value = 1
        # Fixed bonus (b), varying sampling (s)
        sampling_values = [5, 10, 50, 100]
        variants = [f"FETRPOShield_s{s}_b{fixed_value}_t0.275" for s in sampling_values]
        display_names = {
            f"FETRPOShield_s{s}_b{fixed_value}_t0.275": f"Samples = {s}"
            for s in sampling_values
        }
        
        study_title = f"Sampling Ablation (Safety Bonus={fixed_value})"
    else:  # bonus study
        # Fixed sampling (s), varying bonus (b)
        fixed_value = 10
        bonus_values = [0.05, 0.1, 0.5, 1]
        variants = [f"FETRPOShield_s{int(fixed_value)}_b{b}_t0.275" for b in bonus_values]
        display_names = {
            f"FETRPOShield_s{int(fixed_value)}_b{b}_t0.275": f"Safety Bonus = {b}"
            for b in bonus_values
        }
    
        study_title = f"Bonus Ablation (Samples={int(fixed_value)})"
    
    # Replace the viridis colormap with specific colors
    colors = {
        variant: color for variant, color in zip(variants, [
            "#4B0082",  # Indigo
            "#D55E00",  # Orange/Red
            "#009E73",  # Green
            "#CC79A7",  # Pink/Purple
        ])
    }

    # Map simplified names to full environment names and display names
    env_mapping = {
        "Button": f"Safety{robot_type}Button1-v1",
        "Goal": f"Safety{robot_type}Goal1-v1",
        "Push": f"Safety{robot_type}Push1-v1",
        "Circle": f"Safety{robot_type}Circle1-v1",
    }

    # Create display names with robot type prefix
    display_env_names = {
        "Button": f"{robot_type}-Button",
        "Goal": f"{robot_type}-Goal",        
        "Push": f"{robot_type}-Push",
        "Circle": f"{robot_type}-Circle",
    }

    # Filter out excluded environments
    env_order = [env for env in ["Goal", "Button", "Push", "Circle"] 
                 if env not in exclude_envs]
    
    fig, axes = plt.subplots(2, len(env_order), figsize=(12, 6))  # Adjusted figure size for 4 columns

    for env_idx, simple_env_name in enumerate(env_order):
        full_env_name = env_mapping[simple_env_name]
        if full_env_name not in experiments:
            print(f"Warning: No data found for {full_env_name}")
            continue
            
        env_data = experiments[full_env_name]
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx, env_idx]
            
            for variant in variants:
                if variant not in env_data or not env_data[variant]:
                    print(f"Skipping {variant} - no data available")
                    continue

                print(f"Plotting {variant} for {full_env_name} - {metric}")
                
                # Calculate statistics
                metric_key = f"{metric}_Mean"
                
                all_data = pd.concat([df[metric_key] for df in env_data[variant]], axis=1)
                if 'Cost' in metric_key:
                    episode_length = 500 if 'Circle' in simple_env_name or 'Doggo' in simple_env_name else 1000
                    all_data = all_data / episode_length * 100  # Convert to percentage

                mean = all_data.mean(axis=1)
                std = all_data.std(axis=1)
                num_points = len(mean)
                x = np.linspace(0, 2e6, num_points)

                # Plot
                ax.plot(x, mean, label=display_names[variant], color=colors[variant], linewidth=2.5)
                ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=colors[variant])
                
            # Customize axis with bold labels
            if metric_idx == 1:
                ax.set_xlabel("Timesteps (×$10^6$)", fontsize=FONT_SIZES["axis"])
            if env_idx == 0:
                ylabel = "Return" if metric == "EpRet" else "Cost Rate (\%)"
                ax.set_ylabel(ylabel, fontsize=FONT_SIZES["axis"])
            
            if metric_idx == 0:
                # Use the display name with robot type prefix
                ax.set_title(f"\\textbf{{{display_env_names[simple_env_name]}}}", fontsize=FONT_SIZES["title"])

            if metric == "EpRet" and "Push" in simple_env_name:
                ax.set_ylim([-5., 1.])
            elif metric == "EpCost":
                if 'Push' in simple_env_name:
                    ax.set_ylim([-0.1, 5])
                else:
                    ax.set_ylim([-0.1, 10])

            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES["tick"])

            ax.set_xticklabels([str(label.get_text()) for label in ax.get_xticklabels()])
            ax.set_yticklabels([str(label.get_text()) for label in ax.get_yticklabels()])

    # Add legend with bold labels
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        bold_labels = [label for label in labels]
        fig.legend(handles, bold_labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
                  ncol=len(variants), fontsize=FONT_SIZES["legend"], frameon=True)

    plt.tight_layout()
    return fig

def create_ablation_plots(base_path: str, output_dir: str = "plot_ablations") -> None:
    """Create ablation study plots for both sampling and bonus parameters."""
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        experiments = load_and_process_experiments(base_path)
        print(f"Loaded {len(experiments)} experiments")
        
        # Sampling ablation (fixed bonus)
        for robot in ["Point", "Car"]:
            # Fixed bonus = 1.0, varying sampling
            sampling_fig = create_ablation_study_plot_shield(
                experiments, robot, "sampling", fixed_value=1.0,
                exclude_envs=["Doggo"])  # Add exclude_envs parameter
            sampling_output = output_path / f"sampling_ablation_{robot.lower()}.png"
            sampling_fig.savefig(sampling_output, bbox_inches='tight', dpi=300)
            print(f"Saved {robot} sampling ablation plot to: {sampling_output}")
            plt.close(sampling_fig)
            
            # Fixed sampling = 4, varying bonus
            bonus_fig = create_ablation_study_plot_shield(
                experiments, robot, "bonus", fixed_value=4,
                exclude_envs=["Doggo"])  # Add exclude_envs parameter
            bonus_output = output_path / f"bonus_ablation_{robot.lower()}.png"
            bonus_fig.savefig(bonus_output, bbox_inches='tight', dpi=300)
            print(f"Saved {robot} bonus ablation plot to: {bonus_output}")
            plt.close(bonus_fig)
        
    except Exception as e:
        print(f"Error creating/saving plots: {e}")

if __name__ == "__main__":
    base_path = "./results/fe"
    print(f"Looking for logs in: {base_path}")
    create_ablation_plots(base_path)