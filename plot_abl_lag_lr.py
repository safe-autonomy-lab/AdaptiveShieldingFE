import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from typing import Dict, List, Tuple
import matplotlib
from plot_configuration import FONT_SIZES
matplotlib.use("TkAgg")

def parse_experiment_path(path: str) -> Tuple[str, str, str, float, int]:
    """
    Parse experiment path to extract algorithm, environment, algorithm type, learning rate, and seed.
    
    Args:
        path: Path string to parse
        
    Returns:
        Tuple of (algorithm_base, environment, algorithm_type, learning_rate, seed)
    """
    path_str = str(path)
    # Pattern to match paths like .../SafetyCarCircle1-v1/OracleTRPOLag_init0.001_lr0.035/seed1
    # Updated to support both TRPO and PPO
    pattern = r".*/([A-Za-z]+[A-Za-z0-9_.-]+)/([A-Za-z]+)(TRPO|PPO)[A-Za-z]+_init[\d.]+_lr([\d.]+)/seed(\d+)"
    print(f"Trying to match pattern on path: {path_str}")
    
    match = re.search(pattern, path_str)
    if match:
        env = match.group(1)
        alg_base = match.group(2)  # Oracle, FE, or empty for standard
        alg_type = match.group(3)  # TRPO or PPO
        lr = float(match.group(4))
        seed = int(match.group(5))
        print(f"Successfully parsed: env={env}, alg_base={alg_base}, alg_type={alg_type}, lr={lr}, seed={seed}")
        return alg_base, env, alg_type, lr, seed
    
    raise ValueError(f"Invalid path format: {path}")

def calculate_moving_window_stats(csv_path: str, window_size: int = 100) -> pd.DataFrame:
    """
    Calculate moving window statistics from CSV log file.
    
    Args:
        csv_path: Path to the CSV file
        window_size: Size of the moving window
        
    Returns:
        DataFrame with moving window statistics
    """
    df = pd.read_csv(csv_path)
    
    stats = pd.DataFrame()
    stats['total_steps'] = df['total_steps']
    
    for column in ['EpRet', 'EpCost']:
        stats[f'{column}_Mean'] = df[column].rolling(window=window_size, min_periods=1).mean()
        stats[f'{column}_Std'] = df[column].rolling(window=window_size, min_periods=1).std()
    
    return stats

def load_and_process_experiments(base_path: str, representation: str = "oracle", alg_type: str = "TRPO") -> Dict[str, Dict[str, List[pd.DataFrame]]]:
    """
    Load and organize experiment data for learning rate ablation studies.
    
    Args:
        base_path: Base directory path
        representation: Type of representation ("oracle", "fe", or "standard")
        alg_type: Algorithm type ("TRPO" or "PPO")
        
    Returns:
        Nested dictionary containing organized experiment data
    """
    base_dir = Path(base_path) / representation
    experiments: Dict[str, Dict[str, List[pd.DataFrame]]] = {}
    
    print(f"\nDEBUG: Searching in base directory: {base_dir.absolute()}")
    print(f"DEBUG: Directory exists: {base_dir.exists()}")
    
    for log_name in ["history.csv"]:
        print(f"\nLooking for {log_name} files...")
        
        for exp_dir in base_dir.rglob(log_name):
            try:
                print(f"\nProcessing file: {exp_dir}")
                
                try:
                    alg_base, env, found_alg_type, lr, seed = parse_experiment_path(str(exp_dir.parent))
                    # Skip if this is not the algorithm type we're looking for
                    if found_alg_type != alg_type:
                        print(f"Skipping {found_alg_type} algorithm (looking for {alg_type})")
                        continue
                        
                    variant_key = f"{alg_base}{alg_type}Lag_init0.001_lr{lr}"
                    
                    if env not in experiments:
                        experiments[env] = {}
                    if variant_key not in experiments[env]:
                        experiments[env][variant_key] = []
                    
                    df = pd.read_csv(str(exp_dir))
                    processed_df = pd.DataFrame()
                    
                    if 'EpRet' in df.columns and 'EpCost' in df.columns:
                        processed_df = calculate_moving_window_stats(str(exp_dir))
                    elif 'Metrics/EpRet' in df.columns and 'Metrics/EpCost' in df.columns:
                        processed_df['EpRet_Mean'] = df['Metrics/EpRet']
                        processed_df['EpCost_Mean'] = df['Metrics/EpCost']
                    
                    if not processed_df.empty:
                        experiments[env][variant_key].append(processed_df)
                        print(f"Successfully added data for {env}/{variant_key}/seed{seed}")
                
                except ValueError as ve:
                    # This is for paths that don't match our pattern
                    print(f"Skipping file due to parsing error: {ve}")
                    continue
                    
            except Exception as e:
                print(f"Error processing {exp_dir}: {e}")
                import traceback
                print(traceback.format_exc())
    
    # Add this after processing files
    print("\nDEBUG: Processed experiments structure:")
    for env in experiments:
        print(f"\nEnvironment: {env}")
        for variant in experiments[env]:
            print(f"  Variant: {variant}")
            print(f"  Number of seeds: {len(experiments[env][variant])}")

    return experiments

def create_lr_ablation_plot(
    experiments: Dict[str, Dict[str, List[pd.DataFrame]]],
    robot_type: str,
    representation: str,
    alg_type: str,
    metrics: List[str] = ["EpRet", "EpCost"]
) -> plt.Figure:
    """
    Create ablation study plots for Lagrangian learning rates.
    
    Args:
        experiments: Dictionary containing experiment data
        robot_type: Type of robot ("Point", "Car", or "Doggo")
        representation: Type of representation used
        alg_type: Algorithm type ("TRPO" or "PPO")
        metrics: List of metrics to plot
    
    Returns:
        matplotlib Figure object
    """
    # Set up LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
    # sns.set_style("whitegrid")
    
    # Learning rates to analyze
    learning_rates = [0.0035, 0.035, 0.35, 3.5]
    variants = [f"{representation.capitalize()}{alg_type}Lag_init0.001_lr{lr}" for lr in learning_rates]
    display_names = {variant: f"LR = {lr}" for variant, lr in zip(variants, learning_rates)}
    
    colors = {
        variant: color for variant, color in zip(variants, [
            "#E69F00",  # Orange
            "#D55E00",  # Red
            "#009E73",  # Green
            "#CC79A7",  # Pink/Purple
        ])
    }

    # Map environment names and create display names with robot type prefix
    env_mapping = {
        "Button": f"Safety{robot_type}Button1-v1",
        "Goal": f"Safety{robot_type}Goal1-v1",
        "Circle": f"Safety{robot_type}Circle1-v1",
        "Push": f"Safety{robot_type}Push1-v1"
    }
    
    # Create display names with robot type prefix
    display_env_names = {
        "Button": f"{robot_type}-Button",
        "Goal": f"{robot_type}-Goal",
        "Circle": f"{robot_type}-Circle",
        "Push": f"{robot_type}-Push"
    }
    
    env_order = ["Button", "Goal", "Circle", "Push"]
    fig, axes = plt.subplots(2, len(env_order), figsize=(12, 6))

    # Add these debug prints at the start
    print("\nDEBUG: Starting plot creation")
    print(f"DEBUG: Robot type: {robot_type}")
    print(f"DEBUG: Algorithm type: {alg_type}")
    print(f"DEBUG: Available environments: {list(experiments.keys())}")
    print(f"DEBUG: Variants to plot: {variants}")
    
    for env_idx, simple_env_name in enumerate(env_order):
        full_env_name = env_mapping[simple_env_name]
        print(f"\nDEBUG: Processing environment: {full_env_name}")
        if full_env_name not in experiments:
            print(f"Warning: No data found for {full_env_name}")
            continue
            
        env_data = experiments[full_env_name]
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx, env_idx]
            
            # Debug availability of variants for this environment
            for variant in variants:
                print(f"DEBUG: Checking variant {variant} availability: {variant in env_data}")
                if variant in env_data:
                    print(f"DEBUG: Number of seeds for {variant}: {len(env_data[variant])}")
            
            for variant in variants:
                if variant not in env_data or not env_data[variant]:
                    print(f"Skipping {variant} - no data available")
                    continue

                metric_key = f"{metric}_Mean"
                try:
                    all_data = pd.concat([df[metric_key] for df in env_data[variant]], axis=1)
                    if 'Cost' in metric_key:
                        episode_length = 500 if 'Circle' in simple_env_name else 1000
                        all_data = all_data / episode_length * 100  # Convert to percentage

                    mean = all_data.mean(axis=1)
                    std = all_data.std(axis=1)
                    num_points = len(mean)
                    x = np.linspace(0, 2e6, num_points)

                    ax.plot(x, mean, label=display_names[variant], color=colors[variant], linewidth=2.5)
                    ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=colors[variant])
                    
                except Exception as e:
                    print(f"Error plotting {variant}: {e}")

            # Customize axis with LaTeX bold text
            if metric_idx == 1:
                ax.set_xlabel("Timesteps (×$10^6$)", fontsize=FONT_SIZES["axis"])
            if env_idx == 0:
                ylabel = "Return" if metric == "EpRet" else "Cost Rate (\%)"
                ax.set_ylabel(ylabel, fontsize=FONT_SIZES["axis"])
            
            if metric_idx == 0:
                # Use the display name with robot type prefix
                ax.set_title(f"\\textbf{{{display_env_names[simple_env_name]}}}", fontsize=FONT_SIZES["title"])

            if metric == "EpCost":
                ax.set_ylim([-1, 10])

            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES["tick"])
            

    # Add legend with bold labels
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        bold_labels = [label for label in labels]
        fig.legend(handles, bold_labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
                  ncol=len(variants), fontsize=FONT_SIZES["legend"], frameon=True)

    # Adjust figure layout to make room for title and legend
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Add algorithm type to title with adjusted position
    fig.suptitle(f"{alg_type}-Lag Learning Rate Ablation ({representation.capitalize()} Representation)", fontsize=FONT_SIZES["title"])

    return fig

def create_lr_ablation_plots(base_path: str, representation: str = "oracle", output_dir: str = "plot_ablations") -> None:
    """
    Create learning rate ablation plots for all robot types.
    
    Args:
        base_path: Base directory path
        representation: Type of representation to analyze
        output_dir: Directory to save output plots
    """
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    
    # Create plots for TRPO
    trpo_experiments = load_and_process_experiments(base_path, representation, "TRPO")
    print(f"Loaded {len(trpo_experiments)} TRPO experiments")
    
    for robot in ["Point", "Car"]:
        fig = create_lr_ablation_plot(trpo_experiments, robot, representation, "TRPO")
        output_file = output_path / f"lr_ablation_{robot.lower()}_{representation}_trpolag.png"
        fig.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Saved {robot} TRPO learning rate ablation plot to: {output_file}")
        plt.close(fig)
        
    # Create plots for PPO
    ppo_experiments = load_and_process_experiments(base_path, representation, "PPO")
    print(f"Loaded {len(ppo_experiments)} PPO experiments")
    
    for robot in ["Point", "Car"]:
        fig = create_lr_ablation_plot(ppo_experiments, robot, representation, "PPO")
        output_file = output_path / f"lr_ablation_{robot.lower()}_{representation}_ppolag.png"
        fig.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Saved {robot} PPO learning rate ablation plot to: {output_file}")
        plt.close(fig)
        
if __name__ == "__main__":
    base_path = "./results"
    print(f"DEBUG: Base path exists: {Path(base_path).exists()}")
    print(f"DEBUG: Base path contents: {list(Path(base_path).glob('*'))}")
    create_lr_ablation_plots(base_path, representation="oracle")
