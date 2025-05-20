"""Script to generate reward vs. cost trade-off scatter plots comparing density and damping effects."""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from typing import Dict, List, Optional, Tuple
from plot_configuration import COLORS_RQ1, ALGORITHM_MARKERS, FONT_SIZES

# Define the order of environments used in the specific suffix lists
ENV_TYPES_ORDER = ["Goal", "Button", "Push", "Circle"]

# Define specific suffix prefixes (_Num1_) for certain algorithms
# Structure: {algo_base_name: {robot_type: {level: {env_type: suffix_prefix}}}}
SPECIFIC_ALGORITHM_SUFFIXES = {
    "ShieldedPPO": {
        "Car": {
            "2": dict(zip(ENV_TYPES_ORDER, ["_100_", "_10_", "_5_", "_5_"])),
        },
        "Point": {
            "2": dict(zip(ENV_TYPES_ORDER, ["_100_", "_50_", "_5_", "_10_"])),
        },
    },
    "ShieldedTRPO": {
        "Car": {
            "2": dict(zip(ENV_TYPES_ORDER, ["_50_", "_50_", "_10_", "_100_"])),
        },
        "Point": {
            "2": dict(zip(ENV_TYPES_ORDER, ["_10_", "_100_", "_10_", "_5_"])),
        },
    },
}
DEFAULT_SUFFIX_PREFIX = "_10_"  # Default prefix if no specific rule applies

# Regex to capture base algorithm name and the _Num1_Num2 suffix
SUFFIX_PATTERN = re.compile(r"^(.*?)_(\d+)_(\d+)$")



def find_csv_files(
    base_folder: str,
    robot_type: str,
    env_type: str,
    level: str,
    algorithm_pattern: str,
) -> List[str]:
    """Find all relevant evaluation_results.csv files."""
    csv_files = []
    
    # Construct the environment folder name
    env_folder = f"Safety{robot_type}{env_type}{level}"
    env_path = os.path.join(base_folder, env_folder)
    
    if not os.path.isdir(env_path):
        print(f"Warning: Environment folder not found: {env_path}")
        return []
    
    # Compile the algorithm pattern regex
    algo_regex = re.compile(algorithm_pattern)
    
    # Iterate through algorithm folders
    for algo_folder in os.listdir(env_path):
        algo_path = os.path.join(env_path, algo_folder)
        
        # Check if it's a directory and matches the algorithm pattern
        if os.path.isdir(algo_path) and algo_regex.search(algo_folder):
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
    # Regex to capture algorithm name and seed (e.g., seed0, seed1)
    seed_pattern = re.compile(r"^seed(\d+)$")

    for file_path in csv_files:
        try:
            # Extract env_name and algo from the path
            parts = file_path.split(os.sep)
            # Expected path: ./ood_evaluation_folder_damping/EnvName/AlgoName/SeedX/results.csv
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
        return pd.DataFrame()  # Return empty DataFrame if no data

    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    return combined_df


def simplify_algorithm_name(algo_name: str) -> str:
    """Simplify algorithm name for display purposes."""
    # Remove parameters like _10_1 from names
    base_name = re.sub(r'_\d+_\d+$', '', algo_name)
    # Ours
    if "Shield" in base_name and 'ACP' in base_name:
        return "Shield"
    elif "SafetyObjOnly_TRPO" in base_name:
        return "SRO"
    elif "ShieldedTRPO" in base_name and not "no_ACP" in base_name:
        return "SRO + Shield"
    # Baslines
    elif "PPOLag" in base_name and "Shield" not in base_name:
        return "PPO-Lag"
    elif "TRPOLag" in base_name and "Shield" not in base_name:
        return "TRPO-Lag"
    elif "CPO" in base_name:
        return "CPO"
    elif "USL" in base_name:
        return "USL"
    else:
        return "Unknown"
    
    return base_name


def get_algorithm_colors(algorithms: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    """Generate a color map for algorithms."""
    # Use colors from COLORS_RQ1 configuration
    color_dict = {}
    
    for algo in algorithms:
        if "TRPO-Lag" in algo:
            color_dict[algo] = to_rgba(COLORS_RQ1.get("TRPO-Lag", "#ff0000"))
        elif "PPO-Lag" in algo:
            color_dict[algo] = to_rgba(COLORS_RQ1.get("PPO-Lag", "#FFA500"))
        elif "SRO" in algo and "Shield" not in algo:   
            color_dict[algo] = to_rgba(COLORS_RQ1.get("SRO", "#2ca02c"))
        elif "SRO + Shield" in algo:
            color_dict[algo] = to_rgba(COLORS_RQ1.get("SRO + Shield", "#0000ff"))
        elif "Shield" in algo and "TRPO" not in algo:
            color_dict[algo] = to_rgba(COLORS_RQ1.get("Shield", "#32cd32"))
        elif "CPO" in algo:
            color_dict[algo] = to_rgba(COLORS_RQ1.get("CPO", "#CC79A7"))
        elif "USL" in algo:
            color_dict[algo] = to_rgba(COLORS_RQ1.get("USL", "#654321"))
        else:
            continue
            # Default color for any other algorithms
            color_dict[algo] = to_rgba("#949494")  # gray
            
            
    return color_dict


def plot_density_damping_comparison(
    folder: str,
    robot_types: List[str],
    level: str,
    algorithm_pattern: str = ".*",  # Default to match all algorithms
    output_file: Optional[str] = None,
    env_types: List[str] = ["Goal", "Button", "Push", "Circle"],
    reference_type: str = "median",  # Options: "mean", "median", "percentile"
    percentile: float = 75.0,  # If reference_type is "percentile", use this percentile (0-100)
) -> None:
    """
    Create side-by-side comparison plots of density and damping effects.
    
    Args:
        folder: Folder containing evaluation data
        robot_types: List of robot types (Point and Car)
        level: Difficulty level (2 or 3)
        algorithm_pattern: Regex pattern to match algorithm names
        output_file: Path to save the output image
        env_types: List of environment types to include
        reference_type: Type of reference lines to show ("mean", "median", or "percentile")
        percentile: Percentile value to use if reference_type is "percentile" (0-100)
    """
    # Set up the plotting style
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
    })
    
    # Create figure with 2 rows (Point, Car) x 4 columns (environments)
    fig, axes = plt.subplots(2, 4, figsize=(12, 8))
    
    # Store all algorithm display names for consistent legend
    all_algo_display_names = set()
    
    # Process each robot type
    for row_idx, robot_type in enumerate(robot_types):
        # Process each environment type
        for col_idx, env_type in enumerate(env_types):
            # Get the current subplot
            ax = axes[row_idx, col_idx]
            
            # Find CSV files for this environment
            csv_files = find_csv_files(
                base_folder=folder,
                robot_type=robot_type,
                env_type=env_type,
                level=level,
                algorithm_pattern=algorithm_pattern
            )
            
            if not csv_files:
                print(f"No data found for {robot_type}{env_type}{level}")
                ax.text(0.5, 0.5, f"No data for\n{env_type}", 
                        ha='center', va='center', transform=ax.transAxes, fontsize=FONT_SIZES["axis"])
                continue
            
            # Process results (now we keep individual seed data)
            raw_data = process_results(csv_files)
            
            if raw_data.empty:
                print(f"No raw results found for {robot_type}{env_type}{level}")
                ax.text(0.5, 0.5, f"No results for\n{env_type}", 
                        ha='center', va='center', transform=ax.transAxes, fontsize=FONT_SIZES["axis"])
                continue
            
            # --- Filter results based on the specific algorithm suffix rules ---
            print(f"Filtering {robot_type}{env_type}{level} data based on specific rules...")
            
            # Define the filter function for this specific context
            def filter_algorithm(algo_name: str) -> bool:
                # Handle the _10_0 special case first
                if algo_name.endswith("_10_0"):
                    return True  # Always keep _10_0
                
                match = SUFFIX_PATTERN.match(algo_name)
                if not match:
                    # Algorithm doesn't have the _Num_Num suffix pattern, keep it
                    return True
                else:
                    # Extract base name and suffix details
                    base_name, num1_str, num2_str = match.groups()
                    
                    # Check if this is an algorithm with specific rules
                    if base_name in SPECIFIC_ALGORITHM_SUFFIXES:
                        # Get the specific target prefix for this combination
                        target_prefix = (
                            SPECIFIC_ALGORITHM_SUFFIXES.get(base_name, {})
                            .get(robot_type, {})
                            .get(level, {})
                            .get(env_type, DEFAULT_SUFFIX_PREFIX)
                        )
                        
                        # Check if the algorithm's prefix matches the target prefix
                        actual_prefix = f"_{num1_str}_"
                        return actual_prefix == target_prefix
                    else:
                        # For algorithms without specific rules, use default (_10_)
                        actual_prefix = f"_{num1_str}_"
                        return actual_prefix == DEFAULT_SUFFIX_PREFIX
            
            # Apply the filter
            keep_mask = raw_data['algorithm'].apply(filter_algorithm)
            filtered_data = raw_data[keep_mask]
            
            if filtered_data.empty:
                print(f"Warning: No algorithms passed filtering for {robot_type}{env_type}{level}")
                # Provide context on why it might be empty
                specific_rules = []
                for base_name in SPECIFIC_ALGORITHM_SUFFIXES:
                    if (robot_type in SPECIFIC_ALGORITHM_SUFFIXES[base_name] and 
                        level in SPECIFIC_ALGORITHM_SUFFIXES[base_name][robot_type] and
                        env_type in SPECIFIC_ALGORITHM_SUFFIXES[base_name][robot_type][level]):
                        prefix = SPECIFIC_ALGORITHM_SUFFIXES[base_name][robot_type][level][env_type]
                        specific_rules.append(f"{base_name}: {prefix}")
                
                filter_info = "\n".join(specific_rules) if specific_rules else f"Default: {DEFAULT_SUFFIX_PREFIX}"
                
                ax.text(
                    0.5, 0.5, 
                    f"No data after filtering\nRules applied:\n{filter_info}",
                    ha='center', va='center', transform=ax.transAxes, fontsize=FONT_SIZES["axis"]
                )
                continue
            # --- End Filtering --- 

            # Get unique algorithms
            unique_algos = filtered_data["algorithm"].unique()

            # Create data structure to store individual seed data and mean values
            algo_data = {}
            
            # For each algorithm, extract individual seeds and compute mean
            for algo in unique_algos:
                algo_df = filtered_data[filtered_data["algorithm"] == algo]
                simple_name = simplify_algorithm_name(algo)
                
                # Initialize data for this algorithm
                if simple_name not in algo_data:
                    algo_data[simple_name] = {
                        "seeds": [],
                        "mean_reward": 0,
                        "mean_cost": 0,
                        "std_reward": 0,
                        "std_cost": 0
                    }
                
                # Process each seed
                for seed in algo_df["seed"].unique():
                    seed_df = algo_df[algo_df["seed"] == seed]
                    
                    # Get reward and cost for this seed
                    reward_row = seed_df[seed_df["Metric"] == "Average episode reward"]
                    cost_row = seed_df[seed_df["Metric"] == "Average episode cost"]
                    
                    if not reward_row.empty and not cost_row.empty:
                        reward = reward_row["Value"].values[0]
                        # Convert cost to cost rate (%) based on environment type
                        cost = cost_row["Value"].values[0]
                        cost_rate = cost / 500 * 100 if "Circle" in env_type else cost / 1000 * 100
                        
                        # Store seed data
                        algo_data[simple_name]["seeds"].append({
                            "seed": seed,
                            "reward": reward,
                            "cost": cost_rate  # Store cost rate instead of absolute cost
                        })
                
                # Calculate mean and std if we have seed data
                if algo_data[simple_name]["seeds"]:
                    rewards = [s["reward"] for s in algo_data[simple_name]["seeds"]]
                    costs = [s["cost"] for s in algo_data[simple_name]["seeds"]]
                          
                    algo_data[simple_name]["mean_reward"] = np.mean(rewards)
                    algo_data[simple_name]["mean_cost"] = np.mean(costs)
                    algo_data[simple_name]["std_reward"] = np.std(rewards)
                    algo_data[simple_name]["std_cost"] = np.std(costs)
                
                # Add to global set of display names
                all_algo_display_names.add(simple_name)
            
            # Get colors for display names
            color_dict = get_algorithm_colors(list(algo_data.keys()))
            
            # Plot each algorithm with individual seeds and mean
            for algo_name, data in algo_data.items():
                if not data["seeds"]:
                    continue
                    
                if algo_name not in color_dict:
                    continue
                    
                color = color_dict[algo_name]
                marker = ALGORITHM_MARKERS.get(algo_name, "o")
                
                # Plot individual seeds with transparency - commented out to only show means
                # for seed_data in data["seeds"]:
                #     ax.scatter(
                #         seed_data["cost"],
                #         seed_data["reward"],
                #         color=color,
                #         marker=marker,
                #         s=70,  # Smaller size for individual seeds
                #         alpha=0.2,  # Transparency for individual seeds
                #         edgecolors='none',
                #         zorder=1  # Lower zorder to ensure mean is on top
                #     )
                
                # Plot mean value with solid color
                ax.scatter(
                    data["mean_cost"],
                    data["mean_reward"],
                    color=color,
                    marker=marker,
                    s=120,  # Larger size for mean
                    alpha=1.0,  # Solid color for mean
                    edgecolors='black',
                    linewidth=1,
                    label=algo_name,
                    zorder=2  # Higher zorder to place on top
                )
            
            # Add a star marker to indicate desirable performance (high reward, low cost)
            # Find the boundaries of the data
            costs = [data["mean_cost"] for data in algo_data.values() if data["seeds"]]
            rewards = [data["mean_reward"] for data in algo_data.values() if data["seeds"]]
            
            if costs and rewards:
                # Cost values are already converted to rates, so no need for additional conversion
                min_cost, max_cost = min(costs), max(costs)
                min_reward, max_reward = min(rewards), max(rewards)
                
                # Calculate reference values based on selected method
                if reference_type == "mean":
                    ref_cost = np.mean(costs)
                    ref_reward = np.mean(rewards)
                    ref_label = "Mean"
                elif reference_type == "median":
                    ref_cost = np.median(costs)
                    ref_reward = np.median(rewards)
                    ref_label = "Median"
                elif reference_type == "percentile":
                    # For cost, we want the lower percentile (better)
                    # For reward, we want the upper percentile (better)
                    ref_cost = np.percentile(costs, 100 - percentile)
                    ref_reward = np.percentile(rewards, percentile)
                    ref_label = f"{100 - percentile:.0f}th Percentile"
                else:
                    # Default to median if invalid type
                    ref_cost = np.median(costs)
                    ref_reward = np.median(rewards)
                    ref_label = "Median"
                
                # Add dotted lines for reference cost and reward
                # Horizontal line for reference reward
                ax.axhline(
                    y=ref_reward,
                    color='gray',
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=1
                )
                
                # Vertical line for reference cost
                ax.axvline(
                    x=ref_cost,
                    color='gray',
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=1
                )
                
                # Highlight the desirable region (top-left quadrant) with light yellow
                # Create a rectangle patch for the highlighted area
                # The rectangle extends from the minimum cost to the reference cost
                # and from the reference reward to the maximum reward
                cost_pad = (max_cost - min_cost) * 0.1 if max_cost > min_cost else max(0.1, abs(min_cost) * 0.1)
                reward_pad = (max_reward - min_reward) * 0.1 if max_reward > min_reward else max(0.1, max_reward * 0.1)
                highlight_rect = plt.Rectangle(
                    (min_cost - cost_pad, ref_reward),  # Bottom-left corner of rectangle
                    ref_cost + cost_pad - min_cost,     # Width (from min_cost to ref_cost)
                    max_reward + reward_pad - ref_reward,   # Height (from ref_reward to max_reward)
                    facecolor='lightyellow',
                    alpha=1.,
                    zorder=0  # Put behind all other elements
                )
                
                # Add the rectangle to the plot
                ax.add_patch(highlight_rect)
                
                # Place the star at the desirable location (low cost, high reward)
                # Use the top left corner of the data range as the ideal point
                star_cost = min_cost
                star_reward = max_reward
                
                # Add the star marker to indicate desirable performance
                ax.scatter(
                    star_cost,
                    star_reward,
                    marker='*',
                    s=200,  # Large size for visibility
                    color='gold',
                    edgecolors='black',
                    linewidth=1.5,
                    label='Desirable Performance',
                    zorder=3  # Highest zorder to place on top of everything
                )
            
            # Configure subplot
            # if row_idx == 0:
            # ax.set_title(f"\\textbf{{{env_name}}}", fontsize=FONT_SIZES["title"])
            ax.set_title(f"\\textbf{{{robot_type}-{env_type}}}", fontsize=FONT_SIZES["axis"])
            
            # Add row labels to indicate robot type
            if col_idx == 0:
                ax.set_ylabel(f"Return", fontsize=FONT_SIZES["axis"])
            
            # Only add x-axis label to the bottom row
            
            ax.set_xlabel("Cost Rate (\\%)", fontsize=FONT_SIZES["axis"])
            
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set better x and y limits based on data range
            if costs and rewards:
                min_cost, max_cost = min(costs), max(costs)
                min_reward, max_reward = min(rewards), max(rewards)
                
                # Add padding 
                cost_pad = (max_cost - min_cost) * 0.1 if max_cost > min_cost else max(0.1, abs(min_cost) * 0.1)
                reward_pad = (max_reward - min_reward) * 0.1 if max_reward > min_reward else max(0.1, max_reward * 0.1)
                
                # Set x-axis limits but invert the direction (higher values on left)
                ax.set_xlim(min_cost - 1 * cost_pad, max_cost + 1 * cost_pad)
                ax.set_ylim(min_reward - 1 * reward_pad, max_reward + 1 * reward_pad)
                
            else:
                # Invert x-axis direction if no data points
                ax.invert_xaxis()
    
    # Create a unified legend
    # Convert algorithm display names to sorted list for consistent ordering
    all_algo_display_names = list(all_algo_display_names)
    color_dict = get_algorithm_colors(all_algo_display_names)
    
    # Create legend handles manually to ensure correct markers
    legend_handles = []
    legend_labels = []
    
    # Define the preferred order for algorithms in the legend
    baseline_legend_order = ["TRPO-Lag", "PPO-Lag", "CPO", "USL"]
    ours_legend_order = ["Shield", "SRO + Shield", "SRO"]
    
    # Add special elements to the legend
    # Add the star marker to the legend order
    legend_order = baseline_legend_order + ours_legend_order + ["Desirable Performance"]
    
    # Flag to control whether legend is outside or inside plots
    outside_legend = True
    
    if outside_legend:
        # Create legend entries according to preferred order
        for display_name in legend_order:
            if display_name == "Desirable Performance":
                # Special handling for the star marker
                handle = plt.Line2D(
                    [0], [0],
                    marker='*',
                    color='w',
                    markerfacecolor='gold',
                    markeredgecolor='black',
                    markersize=12
                )
                legend_handles.append(handle)
                legend_labels.append(display_name)
            elif display_name in all_algo_display_names:
                color = color_dict[display_name]
                marker = ALGORITHM_MARKERS.get(display_name, "o")
                handle = plt.Line2D(
                    [0], [0], 
                    marker=marker, 
                    color='w', 
                    markerfacecolor=color,
                    markeredgecolor='black',
                    markersize=10
                )
                legend_handles.append(handle)
                legend_labels.append(display_name)
        
        # Add reference lines to the legend
        handle = plt.Line2D(
            [0], [0],
            linestyle='--',
            color='gray',
            linewidth=1.5
        )
        legend_handles.append(handle)
        legend_labels.append(f"{ref_label} Values")
        
        # Separate handles and labels for baselines and our methods
        baseline_handles, baseline_labels = [], []
        ours_handles, ours_labels = [], []
        special_handles, special_labels = [], []

        for handle, label in zip(legend_handles, legend_labels):
            if label in baseline_legend_order:
                baseline_handles.append(handle)
                baseline_labels.append(label)
            elif label in ours_legend_order:
                ours_handles.append(handle)
                ours_labels.append(label)
            else:
                special_handles.append(handle)
                special_labels.append(label)

        y_offset = 1.1
        # Create baseline legend - position at the bottom left
        baseline_legend = fig.legend(
            baseline_handles,
            baseline_labels,
            loc='upper left',
            bbox_to_anchor=(0.03, y_offset),  # Place it at the very bottom left
            ncol=2,  # 2 columns
            frameon=True,
            title="Baselines",
            title_fontsize=FONT_SIZES["legend"],
            fontsize=FONT_SIZES["legend"],
            handletextpad=0.3,
            borderpad=0.4
        )

        
        # For 3 items, add a blank entry to create a 2x2 grid with the 3rd item centered
        # Create a dummy blank handle for the 4th position
        blank_handle = plt.Line2D([], [], color="none", marker="None")
        # Add the empty handle on the right side of the second row (last position)
        modified_handles = ours_handles[:2] + [ours_handles[2], blank_handle]
        modified_labels = ours_labels[:2] + [ours_labels[2], ""]
        
        ours_legend = fig.legend(
            modified_handles,
            modified_labels,
            loc='upper right', 
            bbox_to_anchor=(0.66, y_offset),  # Place it at the very bottom right
            ncol=2,  # 2 columns with 2 rows
            frameon=True,
            title="Our Methods",
            title_fontsize=FONT_SIZES["legend"],
            fontsize=FONT_SIZES["legend"],
            handletextpad=0.3,
            borderpad=0.4
        )
    
        # Create special entries legend - position at the bottom center
        special_legend = fig.legend(
            special_handles,
            special_labels,
            loc='upper center',
            bbox_to_anchor=(0.9, y_offset),  # Place it at the very bottom center
            ncol=1,  # 2 columns side by side
            frameon=True,
            title="Special Entries",
            title_fontsize=FONT_SIZES["legend"],
            fontsize=FONT_SIZES["legend"],
            handletextpad=0.3,
            borderpad=0.4
        )
    else:
        # add cost
        # delete title, x axis cost,
        # label should be on top
        # only main legend
        # Add legend to the bottom right subplot
        row_idx = len(robot_types) - 1
        col_idx = 3  # Add to the third subplot (zero-indexed)
        ax = axes[row_idx, col_idx]
        
        # Create legend entries according to preferred order
        for display_name in legend_order:
            if display_name == "Desirable Performance":
                # Special handling for the star marker
                handle = plt.Line2D(
                    [0], [0],
                    marker='*',
                    color='w',
                    markerfacecolor='gold',
                    markeredgecolor='black',
                    markersize=12
                )
                legend_handles.append(handle)
                legend_labels.append(display_name)
            elif display_name in all_algo_display_names:
                color = color_dict[display_name]
                marker = ALGORITHM_MARKERS.get(display_name, "o")
                handle = plt.Line2D(
                    [0], [0], 
                    marker=marker, 
                    color='w', 
                    markerfacecolor=color,
                    markeredgecolor='black',
                    markersize=8
                )
                legend_handles.append(handle)
                legend_labels.append(display_name)
        
        # Get the reference label based on the current reference_type
        if reference_type == "mean":
            ref_line_label = "Mean Values"
        elif reference_type == "median":
            ref_line_label = "Median Values" 
        elif reference_type == "percentile":
            ref_line_label = f"{100 - percentile:.0f}th Percentile"
        else:
            ref_line_label = "Reference Values"
        
        # Add reference lines to the legend
        handle = plt.Line2D(
            [0], [0],
            linestyle='--',
            color='gray',
            linewidth=1.5
        )
        legend_handles.append(handle)
        legend_labels.append(ref_line_label)
        
        # Add the legend to the subplot
        ax.legend(
            legend_handles,
            legend_labels,
            loc='lower right',  # Position in the lower right corner
            frameon=True,
            fontsize=4,  # Small font size to fit in the plot
            framealpha=0.9,
            labelspacing=0.9,  # Increase vertical spacing between legend items
            handletextpad=0.3,  # Space between handle and text
            borderpad=0.3,  # Padding inside legend border
            columnspacing=0.9  # Horizontal space between columns
        )
    
    # Adjust layout
    plt.tight_layout()
    # Add significantly more room at the bottom for the legends and increase space between rows
    plt.subplots_adjust(bottom=0.32, top=0.92, hspace=0.5)
    
    # Save plot if output file is specified
    output_dir = "./plot_main_figures"
    if output_file:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, output_file), dpi=300, bbox_inches="tight")
        print(f"Plot saved to {os.path.join(output_dir, output_file)}")


def main():
    """Main function to create density vs. damping comparison plots."""
    folder = "ood_evaluation_folder"

    # Use a simple pattern to find all algorithm folders initially.
    # Filtering will happen inside plot_density_damping_comparison based on specific rules.
    algorithm_pattern = ".*" 
    
    # Set the reference type for the plots: "mean", "median", or "percentile"
    reference_type = "percentile"  # Change this to select different reference types
    # reference_type = "mean"  # Change this to select different reference types
    percentile = 25.0  # Use 75th percentile (adjust as needed)

    # Define robot types for the combined plot
    robot_types = ["Point", "Car"]

    # Create plots for each level
    for level in ["2"]:
        output_file = f"rq2_level{level}.png"
        print(f"\nProcessing Level {level} Comparison (Both Robot Types)")
        
        plot_density_damping_comparison(
            folder=folder,
            robot_types=robot_types,
            level=level,
            algorithm_pattern=algorithm_pattern, 
            output_file=output_file,
            reference_type=reference_type,
            percentile=percentile,
        )


if __name__ == "__main__":
    main() 