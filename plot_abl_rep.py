import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import matplotlib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import font sizes from plot_configuration
try:
    from plot_configuration import FONT_SIZES
except ImportError:
    # Default font sizes if import fails
    FONT_SIZES = {
        "title": 16,
        "axis": 14,
        "legend": 12,
        "tick": 10
    }
    logger.warning("Could not import FONT_SIZES from plot_configuration, using defaults")

matplotlib.use('TkAgg')


def create_oracle_comparison_plot(
    base_path: str,
    robot_type: str,
    environments: List[str],
    algorithms: List[str] = ["TRPOLag", "PPOLag"],
    metrics: List[str] = ["EpRet", "EpCost"],
    every_n_episode: int = 20
) -> plt.Figure:
    """Create comparison plots between different implementations for a specific robot type."""
    # Set up LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
    
    base_path = Path(base_path)
    logger.info(f"Base path: {base_path}")
    
    # Add directory existence check
    if not base_path.exists():
        raise ValueError(f"Base path does not exist: {base_path}")
    
    # Filter out Run environment
    environments = [env for env in environments if env != "Run"]
    
    # Setup plot
    fig, axes = plt.subplots(2, len(environments), figsize=(12, 6))
    if len(environments) == 1:
        axes = axes.reshape(-1, 1)

    # Define line styles for each algorithm and implementation type
    line_configs = {
        # TRPO variants - Red color family
        "TRPOLag": {
            "oracle": {"color": "#FF0000", "linestyle": "-", "linewidth": 2.5, "label": "Oracle-TRPOLag"},  # Red, solid, thick
            "fe": {"color": "#DD0000", "linestyle": "--", "linewidth": 2.0, "label": "FE-TRPOLag"},  # Darker red, dashed
        },
        # PPO variants - Orange color family
        "PPOLag": {
            "oracle": {"color": "#FFA500", "linestyle": "-", "linewidth": 2.5, "label": "Oracle-PPOLag"},  # Orange, solid, thick
            "fe": {"color": "#DDA500", "linestyle": "--", "linewidth": 2.0, "label": "FE-PPOLag"},  # Darker orange, dashed
        }
    }
    
    env_mapping = {
        env: f"Safety{robot_type}{env}1-v1" for env in environments
    }
    
    # Create display names with robot type prefix
    display_env_names = {
        env: f"{robot_type}-{env}" for env in environments
    }
    
    for env_idx, simple_env_name in enumerate(environments):
        logger.info(f"\nProcessing {robot_type} environment: {simple_env_name}")
        full_env_name = env_mapping[simple_env_name]

        for algo in algorithms:
            logger.info(f"Processing algorithm: {algo}")
            
            for impl_type in ["oracle", "fe"]:
                data_path = base_path / impl_type / full_env_name
                logger.info(f"Checking path: {data_path}")
                
                if not data_path.exists():
                    logger.warning(f"No data found for {data_path}")
                    continue

                logger.info(f"Found directory: {data_path}")
                
                # Load all data
                all_data = {metric: [] for metric in metrics}
                
                # Define all possible patterns for each implementation type
                if impl_type == "oracle":
                    patterns = [f"Oracle{algo}_init0.001_lr0.035"]
                elif impl_type == "fe":
                    patterns = [f"FE{algo}_init0.001_lr0.035"]
                else:  # standard (Shield)
                    patterns = [f"{algo}_init0.001_lr0.035"]

                # Try each pattern
                algo_dirs = []
                for pattern in patterns:
                    logger.info(f"Looking for pattern: {pattern}")
                    found_dirs = list(data_path.glob(f"{pattern}*"))
                    if found_dirs:
                        logger.info(f"Found directories for pattern {pattern}: {found_dirs}")
                        algo_dirs = found_dirs
                        break

                if not algo_dirs:
                    logger.warning(f"No matching directories found for {impl_type} {algo}")
                    continue

                for algo_dir in algo_dirs:
                    # Get all seed directories
                    seed_dirs = list(algo_dir.glob("seed*"))
                    logger.info(f"Found {len(seed_dirs)} seed directories")
                    
                    for seed_dir in seed_dirs:
                        try:
                            # Try both history.csv and progress.csv
                            for log_file in ["history.csv", "progress.csv"]:
                                log_path = seed_dir / log_file
                                if log_path.exists():
                                    logger.info(f"Found log file: {log_path}")
                                    df = pd.read_csv(log_path)
                                    logger.info(f"Loaded CSV with shape: {df.shape}")

                                    # Process metrics
                                    for metric in metrics:
                                        if metric == "EpRet":
                                            metric_col = next((col for col in ["Metrics/EpRet", "EpRet"] 
                                                            if col in df.columns), None)
                                        else:  # EpCost
                                            metric_col = next((col for col in ["Metrics/EpCost", "EpCost"] 
                                                            if col in df.columns), None)
                                        
                                        if metric_col:
                                            logger.info(f"Found metric column: {metric_col}")
                                            data = df[metric_col].values
                                            if 'Cost' in metric:
                                                episode_length = 500 if 'Circle' in simple_env_name else 1000
                                                data = data / episode_length * 100
                                            all_data[metric].append(data)
                                            logger.info(f"Added {metric} data with length {len(data)}")
                                    break  # If we found and processed one log file, skip the other
                        except Exception as e:
                            logger.error(f"Error processing {seed_dir}: {e}", exc_info=True)

                # Plot metrics
                for metric_idx, metric in enumerate(metrics):
                    ax = axes[metric_idx, env_idx]
                    
                    if all_data[metric]:
                        logger.info(f"Plotting {metric} for {impl_type} {algo}")
                        logger.info(f"Number of seeds: {len(all_data[metric])}")
                        data_array = np.array(all_data[metric])
                        logger.info(f"Data array shape: {data_array.shape}")
                        
                        # Handle different length data arrays
                        min_length = min(len(data) for data in all_data[metric])
                        truncated_data = [data[:min_length] for data in all_data[metric]]
                        data_array = np.array(truncated_data)
                        
                        mean = np.mean(data_array, axis=0)
                        std = np.std(data_array, axis=0)
                        num_points = len(mean)
                        x = np.linspace(0, 2e6, num_points)
                        
                        # Get style config for this algorithm/implementation combo
                        config = line_configs[algo][impl_type]
                        logger.info(f"Using style config: {config}")
                        
                        # Plot with the specified style
                        ax.plot(x, mean, 
                                label=config["label"], 
                                color=config["color"], 
                                linestyle=config["linestyle"],
                                linewidth=config["linewidth"])
                        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=config["color"])
                        logger.info(f"Successfully plotted {config['label']}")
                    else:
                        logger.warning(f"No data found for {metric} in {impl_type} {algo}")

                    # Customize axis with LaTeX bold text
                    if metric_idx == 1:
                        ax.set_xlabel("Timesteps (×$10^6$)", fontsize=FONT_SIZES["axis"])
                    if env_idx == 0:
                        ylabel = "Return" if metric == "EpRet" else "Cost Rate (\%)"
                        ax.set_ylabel(ylabel, fontsize=FONT_SIZES["axis"])
                    
                    if metric_idx == 0:
                        ax.set_title(f"\\textbf{{{display_env_names[simple_env_name]}}}", fontsize=FONT_SIZES["title"])

                    if metric == "EpCost":
                        ax.set_ylim([-1, 10])

                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES["tick"])

    # Add legend with handles sorted in a specific order
    handles, labels = [], []
    for ax in axes.flatten():
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    
    # Predefined order
    ordered_labels = ["Oracle-TRPOLag", "FE-TRPOLag", 
                     "Oracle-PPOLag", "FE-PPOLag"]
    ordered_handles = []
    ordered_final_labels = []
    
    for label in ordered_labels:
        if label in labels:
            idx = labels.index(label)
            ordered_handles.append(handles[idx])
            ordered_final_labels.append(label)
    
    if ordered_handles:
        fig.legend(ordered_handles, ordered_final_labels, loc='upper center', 
                  bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=FONT_SIZES["legend"], frameon=True)
    else:
        logger.warning("No legend handles found. This means no data was successfully plotted.")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"Representation Comparison", fontsize=FONT_SIZES["title"])
    return fig

def create_all_comparisons(base_path: str, output_dir: str = "plot_ablations") -> None:
    """Create comparison plots for specified robot types."""
    base_path = Path(base_path)
    logger.info(f"Base path for results: {base_path}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Only generate one plot per robot type
    for robot_type in ["Car", "Point"]:
        environments = ["Goal", "Button", "Push", "Circle"]
        algorithms = ["TRPOLag", "PPOLag"]  

        try:
            # Create single plot combining both algorithms for this robot type
            fig = create_oracle_comparison_plot(
                base_path=base_path,
                robot_type=robot_type,
                environments=environments,
                algorithms=algorithms
            )
        
            output_file = output_path / f"rep_comparison_{robot_type.lower()}.png"
            fig.savefig(output_file, bbox_inches='tight', dpi=300)
            logger.info(f"Saved comparison plot to: {output_file}")
            plt.close(fig)

        except Exception as e:
            logger.error(f"Error creating/saving plots: {e}", exc_info=True)

if __name__ == "__main__":
    base_path = "./results"
    
    # Verify the path exists
    if not Path(base_path).exists():
        logger.error(f"Error: Base path does not exist: {base_path}")
        exit(1)
        
    # Print the contents of the results directory
    logger.info("\nContents of results directory:")
    for item in Path(base_path).iterdir():
        logger.info(f"  - {item}")
    
    create_all_comparisons(base_path) 