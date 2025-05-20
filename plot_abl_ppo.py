import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from typing import Dict, List, Tuple
from plot_configuration import COLORS_RQ1, FONT_SIZES

# Set up LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
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
    # Updated pattern to match all three types of algorithm directories
    pattern = r".*/([A-Za-z]+[A-Za-z0-9_.-]+)/([A-Za-z]+(?:noACP)?(?:_[A-Za-z]+)?(?:Shield)?(?:Only)?(?:_s\d+_b[\d.]+_t[\d.]+)?(?:_init[\d.]+)?(?:_lr[\d.]+)?)/(?:seed(\d+)|$)"
    print(f"Trying to match pattern on path: {path_str}")
    match = re.search(pattern, path_str)
    if match:
        env = match.group(1)
        alg = match.group(2)
        seed = int(match.group(3))
        print(f"Successfully parsed: env={env}, alg={alg}, seed={seed}")
        return alg, env, seed

    else:
        print(f"No match found for path: {path_str}")
        return None, None, None

def load_and_process_experiments(base_path: str) -> Dict[str, Dict[str, List[pd.DataFrame]]]:
    """Load and organize experiment data."""
    base_dir = Path(base_path)
    base_dir_standard = Path(base_path + '/standard')
    base_dir_oracle = Path(base_path + '/oracle')
    base_dir_fe = Path(base_path + '/fe')

    experiments: Dict[str, Dict[str, List[pd.DataFrame]]] = {}
    
    # Add debug print to see what we're looking for
    print(f"\nSearching in base directory: {base_dir.absolute()}")
    
    # Find all logger.csv files recursively
    for log_name in ["logger.csv", "progress.csv", "history.csv"]:
        print(f"\nLooking for {log_name} files...")
        for exp_dir in list(base_dir_standard.rglob(log_name)) + list(base_dir_oracle.rglob(log_name)) + list(base_dir_fe.rglob(log_name)):
            if 'standard' in str(exp_dir) and 'oracle' in str(exp_dir):
                continue
            
            alg, env, seed = parse_experiment_path(str(exp_dir.parent))
            if alg is None:
                continue
            # print(f"Parsed: env={env}, alg={alg}, seed={seed}")
            
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
                
    for env in experiments:
        for alg in experiments[env]:
            print(f"  Algorithm: {alg} - {len(experiments[env][alg])} seeds")
            
    return experiments

def create_mixed_environment_plot(
    experiments: Dict[str, Dict[str, List[pd.DataFrame]]],
    robot_type: str,
    metrics: List[str] = ["EpRet", "EpCost"],
    every_n_episode: int = 20
) -> plt.Figure:
    """Create a 4x2 grid of plots for specific environments from different robots."""
    # Set up LaTeX rendering
    # plt.rcParams.update()
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
    })
    
    baselines = ["OracleTRPOLag_init0.001_lr0.035", "OraclePPOLag_init0.001_lr0.035", "OracleCPO", "OracleUSL"]
    baselines = ["OraclePPOLag_init0.001_lr0.035"]
    ours = ["FEPPOShieldACP", "FEPPOSafetyObjOnly"]
    
    # Define environment-specific algorithms
    ENV_SPECIFIC_ALGORITHMS = {
        # Point Robot
        "Point-Goal": baselines + ours + ["FEPPOShield_s5_b1_t0.275"], 
        "Point-Button": baselines + ours + ["FEPPOShield_s5_b1_t0.275"],
        "Point-Push": baselines + ours + ["FEPPOShield_s10_b0.1_t0.275"],
        "Point-Circle": baselines + ours + ["FEPPOShield_s100_b1_t0.275"],
        # "Point-Goal2": baselines + ["FETRPOShield_s10_b1_t0.275"],
        # Car Robot
        "Car-Goal": baselines + ours + ["FEPPOShield_s5_b1_t0.275"],
        "Car-Button": baselines + ours + ["FEPPOShield_s5_b1_t0.275"],
        "Car-Push": baselines + ours + ["FEPPOShield_s10_b0.1_t0.275"],
        "Car-Circle": baselines + ours + ["FEPPOShield_s100_b1_t0.275"],        
        # "Car-Goal2": baselines + ["FETRPOShield_s5_b1_t0.275"],
        # Doggo Robot
        # "Doggo-Run": baselines + ["Shield_s10_b1_t0.275"]
    }

    VARIATION_LABELS = {
    # baselines
    "OraclePPOLag_init0.001_lr0.035": "PPO-Lag",
    "OracleTRPOLag_init0.001_lr0.035": "TRPO-Lag",
    "OracleCPO": "CPO",
    "OracleUSL": "USL",
    # ours
    "FEPPOShieldACP": "Shield",
    "FEPPOSafetyObjOnly": "SRO",
    # "FEnoACP_PPOShield_s{sampling_nbr}_b{safety_bonus}_t0.275": "PPO-SRO + Shield",
    "FEPPOShield_s{sampling_nbr}_b{safety_bonus}_t0.275": "SRO + Shield",
    }

    # Group labels for legend
    BASELINE_GROUP = ["PPO-Lag"]
    OURS_GROUP = ["Shield", "SRO", "SRO + Shield"]

    COLORS = COLORS_RQ1
    
    # Map environment names
    env_mapping = {
        "Point-Goal": "SafetyPointGoal1-v1",
        "Point-Button": "SafetyPointButton1-v1",
        "Point-Push": "SafetyPointPush1-v1",
        "Point-Circle": "SafetyPointCircle1-v1",
        "Car-Goal": "SafetyCarGoal1-v1",
        "Car-Button": "SafetyCarButton1-v1",
        "Car-Push": "SafetyCarPush1-v1",
        "Car-Circle": "SafetyCarCircle1-v1",
    }
    
    # Define a smaller set of environments to make the plot more manageable
    # You can adjust this to focus on the most interesting environments
    point_env_order = ["Point-Goal", "Point-Button", "Point-Push", "Point-Circle"]
    car_env_order = ["Car-Goal", "Car-Button", "Car-Push", "Car-Circle"]
    if robot_type == "point":
        env_order = point_env_order
    elif robot_type == "car":
        env_order = car_env_order
    else:
        raise ValueError(f"Invalid robot type: {robot_type}")
    
    # Create a 2x2 grid (2 rows for metrics, 2 columns for environments)
    fig, axes = plt.subplots(2, len(env_order), figsize=(12 * len(env_order) // 4, 6))
    
    # Total number of timesteps (2 million)
    total_timesteps = 2e6
    
    for env_idx, env_name in enumerate(env_order):
        full_env_name = env_mapping[env_name]
        if full_env_name not in experiments:
            print(f"Warning: No data found for {full_env_name}")
            continue
            
        env_data = experiments[full_env_name]
        current_algs = ENV_SPECIFIC_ALGORITHMS[env_name]
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx, env_idx]
            
            for alg in current_algs:
                if alg not in env_data or not env_data[alg]:
                    print(f"Skipping {alg} - no data available")
                    continue
                    
                label_alg = alg.split('_')[0]
                print(f"Plotting {alg} for {full_env_name} - {metric}")
                
                # Calculate statistics
                metric_key = f"{metric}_Mean"
                all_data = pd.concat([df[metric_key] for df in env_data[alg]], axis=1)
                if 'Cost' in metric_key:
                    episode_length = 500 if 'Circle' in env_name or 'Doggo' in env_name else 1000
                    all_data = all_data / episode_length * 100  # Convert to percentage
    
                if alg in ["OracleUSL", "OracleFAC"]:
                    mean = all_data.mean(axis=1)
                    std = all_data.std(axis=1)

                    # Downsample if needed
                    step = 2 * every_n_episode if 'Circle' in env_name or 'Doggo' in env_name else every_n_episode
                    mean = mean[::step]
                    std = std[::step]
                    
                    # Calculate timesteps instead of epochs
                    num_points = len(mean)
                    x = np.linspace(0, total_timesteps, num_points)
                else:
                    mean = all_data.mean(axis=1)
                    std = all_data.std(axis=1)
                    # Calculate timesteps instead of epochs
                    num_points = len(mean)
                    x = np.linspace(0, total_timesteps, num_points)

                base_variation = alg
                if "_s" in alg and "_b" in alg and "_t" in alg:
                    # For variations with hyperparameters in the name
                    parts = alg.split("_s")
                    base_variation = parts[0] + "_s{sampling_nbr}_b{safety_bonus}_t0.275"
    
                label_alg = VARIATION_LABELS.get(base_variation, alg)
                ax.plot(x, mean, label=label_alg, color=COLORS[label_alg], linewidth=2.5)
                ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=COLORS[label_alg])
            
            # Format x-axis to show in millions
            ax.xaxis.set_major_formatter(lambda x, pos: f'{x/1e6:.1f}')
            
            # Add y-axis labels only to the left column
            if metric_idx == 0 and env_idx == 0:
                ylabel = "Return" 
                ax.set_ylabel(ylabel, fontsize=FONT_SIZES["axis"])
            elif metric_idx == 1 and env_idx == 0:  # Add Cost label to the first row
                ylabel = "Cost Rate (\\%)"
                ax.set_ylabel(ylabel, fontsize=FONT_SIZES["axis"])

            if metric_idx == 1:
                ax.set_xlabel("Timesteps (×$10^6$)", fontsize=FONT_SIZES["axis"])
            
            # Add environment titles to each row
            if metric_idx == 0:
                ax.set_title(f"\\textbf{{{env_name}}}", fontsize=FONT_SIZES["title"])

            # Set y-axis limits
            if metric == "EpRet" and "Push" in env_name:
                ax.set_ylim([-5., 1.])
            elif metric == "EpCost":
                if 'Push' in env_name:
                    ax.set_ylim([-0.1, 5])
                else:
                    ax.set_ylim([-0.1, 10])

            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES["tick"])

    # Create legend with two rows - one for baselines and one for our methods
    baseline_handles = []
    baseline_labels = []
    ours_handles = []
    ours_labels = []
    
    # Get all handles and labels from the first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    # Separate baselines and our methods
    for h, l in zip(handles, labels):
        if l in BASELINE_GROUP:
            baseline_handles.append(h)
            baseline_labels.append(l)
        elif l in OURS_GROUP:
            ours_handles.append(h)
            ours_labels.append(l)
    
    # Create a Legend for the baselines group
    offset = 0.
    # Create a vertical line separator between baselines and our methods
    baseline_legend = fig.legend(
        baseline_handles + ours_handles,
        baseline_labels + ours_labels,
        loc='upper center',
        bbox_to_anchor=(0.52, 0.02),
        ncol=8,
        fontsize=FONT_SIZES["legend"],
        frameon=True,
        columnspacing=1.0,  # Increase spacing between columns
        handletextpad=0.5,  # Adjust spacing between handle and text
    )

    # Add a vertical line after the 4th column
    legend_verts = baseline_legend.get_bbox_to_anchor().transformed(fig.transFigure)
    line_x = legend_verts.x0 + (legend_verts.width * 4/8)  # Position after 4th column
    line = plt.Line2D([line_x, line_x], 
                     [legend_verts.y0, legend_verts.y1],
                     transform=fig.transFigure,
                     color='gray',
                     linestyle='-',
                     linewidth=1)
    fig.add_artist(line)
    fig.add_artist(baseline_legend)

    plt.tight_layout()
    # Add more room at the top for the two-row legend
    plt.subplots_adjust(top=0.94)
    return fig

def create_comparison_plots(robot_type: str, base_path: str, output_dir: str = "plot_ablations") -> None:
    """Create a single comparison plot with mixed environments."""
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    experiments = load_and_process_experiments(base_path)
    
    # Create and save mixed environment plot
    mixed_fig = create_mixed_environment_plot(experiments, robot_type=robot_type)
    mixed_output = output_path / f"abl_ppo_{robot_type}.png"
    mixed_fig.savefig(mixed_output, bbox_inches='tight', dpi=300)
    print(f"Saved mixed environments plot to: {mixed_output}")
    plt.close(mixed_fig)

if __name__ == "__main__":
    for robot_type in ['point', 'car']:
        base_path = "./results/"
        print(f"Looking for logs in: {base_path}")
        create_comparison_plots(robot_type=robot_type, base_path=base_path)