"""Script to process and plot Average run time for specific evaluations."""

import argparse
import os
import re
from typing import Dict, List, Set
import pandas as pd


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process and generate tables for OOD evaluation results."
    )
    parser.add_argument(
        "--base_folder",
        type=str,
        default="./ood_evaluation_folder",
        help="Base folder containing the evaluation results.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="runtime_table.tex",
        help="Path to save the LaTeX table file.",
    )
    return parser.parse_args()


TARGET_ALGORITHMS = {
    "PPOLag_10_0",
    "TRPOLag_10_0",
    "ShieldedPPO_100_1",
    "ShieldedTRPO_100_1",
}
RUNTIME_METRIC = "Average episode run time"
SHIELD_METRIC = "Average shield triggered"


def find_csv_files(
    base_folder: str,
    allowed_algorithms: Set[str]
) -> List[str]:
    """Find evaluation_results.csv files for environments ending in '2'."""
    csv_files = []
    # Regex to match environments ending in '2' (e.g., SafetyCarCircle2)
    env_pattern = re.compile(r"^Safety(Point|Car)[a-zA-Z]+2$")
    # Regex to capture seed folder (e.g., seed0, seed1)
    seed_pattern = re.compile(r"^seed(\d+)$")

    if not os.path.isdir(base_folder):
        print(f"Error: Base folder not found: {base_folder}")
        return []

    for env_name in os.listdir(base_folder):
        env_path = os.path.join(base_folder, env_name)
        # Check if it's a directory and matches the environment pattern
        if os.path.isdir(env_path) and env_pattern.match(env_name):
            for algo_folder in os.listdir(env_path):
                # Filter algorithms
                if algo_folder not in allowed_algorithms:
                    continue

                algo_path = os.path.join(env_path, algo_folder)
                if os.path.isdir(algo_path):
                    for seed_folder in os.listdir(algo_path):
                        seed_path = os.path.join(algo_path, seed_folder)
                        if os.path.isdir(seed_path) and seed_pattern.match(seed_folder):
                            result_file = os.path.join(
                                seed_path, "evaluation_results.csv"
                            )
                            if os.path.isfile(result_file):
                                csv_files.append(result_file)
    return csv_files


def process_metrics(csv_files: List[str]) -> pd.DataFrame:
    """Read CSV files, filter for runtime and shield triggers, parse info, and aggregate."""
    all_data = []
    seed_pattern = re.compile(r"^seed(\d+)$")
    target_metrics = {RUNTIME_METRIC, SHIELD_METRIC}

    for file_path in csv_files:
        try:
            parts = file_path.split(os.sep)
            # Expected path: ./base_folder/EnvName/AlgoName/SeedX/results.csv
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

                # Filter for the specific metrics immediately
                filtered_df = df[df["Metric"].isin(target_metrics)].copy()

                if not filtered_df.empty:
                    filtered_df["environment"] = env_name
                    filtered_df["algorithm"] = algorithm_name
                    filtered_df["seed"] = seed
                    
                    # Extract robot type (Point or Car)
                    robot_type_match = re.search(r'Safety(Point|Car)', env_name)
                    if robot_type_match:
                        filtered_df["robot_type"] = robot_type_match.group(1)
                    else:
                        filtered_df["robot_type"] = "Unknown"
                    
                    # Extract environment type (Goal, Button, Push, Circle)
                    env_type_match = re.search(r'(Goal|Button|Push|Circle)', env_name)
                    if env_type_match:
                        filtered_df["env_type"] = env_type_match.group(0)
                    else:
                        filtered_df["env_type"] = "Unknown"
                    all_data.append(filtered_df)
            else:
                print(f"Warning: Unexpected file path structure: {file_path}")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def process_raw_data(raw_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Process raw data to extract detailed statistics separated by robot type."""
    if raw_data.empty:
        return {}
    
    metrics_dict = {}
    
    # Process data for each robot type separately
    for robot_type in ["Car", "Point"]:
        robot_data = raw_data[raw_data["robot_type"] == robot_type]
        
        if robot_data.empty:
            continue
            
        # Split data by metric
        runtime_data = robot_data[robot_data["Metric"] == RUNTIME_METRIC]
        shield_data = robot_data[robot_data["Metric"] == SHIELD_METRIC]
        
        # Process runtime data
        if not runtime_data.empty:
            # Group by environment type and algorithm to get stats across seeds
            runtime_stats = runtime_data.groupby(["env_type", "algorithm"])["Value"].agg([
                ("mean", "mean"),
                ("std", "std")
            ]).reset_index()
            metrics_dict[f"{robot_type}_runtime"] = runtime_stats
        
        # Process shield data
        if not shield_data.empty:
            # Group by environment type and algorithm to get stats across seeds
            shield_stats = shield_data.groupby(["env_type", "algorithm"])["Value"].agg([
                ("mean", "mean"),
                ("std", "std")
            ]).reset_index()
            metrics_dict[f"{robot_type}_shield"] = shield_stats
    
    return metrics_dict


def generate_latex_table(
    metrics_dict: Dict[str, pd.DataFrame],
    output_file: str
) -> None:
    """Generate a LaTeX table with separate sections for Car and Point robots."""
    if not metrics_dict:
        print("No data to generate tables.")
        return
    
    # Environment types in desired order
    ENV_TYPES = ["Goal", "Button", "Push", "Circle"]
    ROBOT_TYPES = ["Car", "Point"]
    
    # Filter and order environments
    available_env_types = set()
    for df_key in metrics_dict.keys():
        if "_runtime" in df_key or "_shield" in df_key:
            df = metrics_dict[df_key]
            if "env_type" in df.columns:
                available_env_types.update(df["env_type"].unique())
    
    env_types_to_include = [env for env in ENV_TYPES if env in available_env_types]
    
    # Start building LaTeX content for a standalone document
    latex_content = []
    latex_content.append(r"\documentclass{article}")
    latex_content.append(r"\usepackage{multirow}")  # Add multirow package
    latex_content.append(r"\usepackage{booktabs}")
    latex_content.append(r"\usepackage{siunitx}")
    latex_content.append(r"\begin{document}")
    
    # The actual table
    latex_content.append(r"\begin{table}[t]")
    latex_content.append(r"\centering")
    latex_content.append(r"\begin{tabular}{|l|l|c|c|c|c|}")
    latex_content.append(r"\hline")
    
    # Header row
    header = [r"\textbf{Robots}", r"\textbf{Metrics}"]
    for env_type in env_types_to_include:
        header.append(rf"\textbf{{{env_type}}}")
    latex_content.append(" & ".join(header) + r" \\")
    latex_content.append(r"\hline")
    
    # Process each robot type
    for robot_type in ROBOT_TYPES:
        runtime_df = metrics_dict.get(f"{robot_type}_runtime", pd.DataFrame())
        shield_df = metrics_dict.get(f"{robot_type}_shield", pd.DataFrame())
        
        if runtime_df.empty and shield_df.empty:
            continue
            
        # First row with robot type using multirow - on its own line
        latex_content.append(rf"\multirow{{3}}{{*}}{{{robot_type}}}")
        
        # Row 1: TRPO-Lag runtime - starting with & to align properly
        if not runtime_df.empty:
            row_data = ["& TRPO-Lag (s)"]
            for env_type in env_types_to_include:
                trpolag_data = runtime_df[(runtime_df["env_type"] == env_type) & 
                                         (runtime_df["algorithm"] == "TRPOLag_10_0")]
                if not trpolag_data.empty:
                    mean = trpolag_data["mean"].iloc[0]
                    std = trpolag_data["std"].iloc[0]
                    row_data.append(f"{mean:.2f}$\\pm${std:.2f}")
                else:
                    row_data.append("-")
            latex_content.append(" & ".join(row_data) + r" \\")
        
        # Row 2: TRPO-Safe + Shield + ACP (Ours) runtime
        if not runtime_df.empty:
            row_data = ["& Ours (s)"]
            for env_type in env_types_to_include:
                trposhield_data = runtime_df[(runtime_df["env_type"] == env_type) & 
                                            (runtime_df["algorithm"] == "ShieldedTRPO_100_1")]
                if not trposhield_data.empty:
                    mean = trposhield_data["mean"].iloc[0]
                    std = trposhield_data["std"].iloc[0]
                    row_data.append(f"{mean:.2f}$\\pm${std:.2f}")
                else:
                    row_data.append("-")
            latex_content.append(" & ".join(row_data) + r" \\")
        
        # Row 3: Shield Triggers
        if not shield_df.empty:
            row_data = ["& Shield Triggers (\\%)"]
            for env_type in env_types_to_include:
                episode_length = 500 if env_type == "Circle" else 1000
                shield_trigger_data = shield_df[(shield_df["env_type"] == env_type) & 
                                               (shield_df["algorithm"] == "ShieldedTRPO_100_1")]
                if not shield_trigger_data.empty:
                    # Convert to percentage (divide by episode length and multiply by 100)
                    mean = (shield_trigger_data["mean"].iloc[0] / episode_length) * 100
                    std = (shield_trigger_data["std"].iloc[0] / episode_length) * 100
                    row_data.append(f"{mean:.2f}$\\pm${std:.2f}")
                else:
                    row_data.append("-")
            latex_content.append(" & ".join(row_data) + r" \\")
        
        # Add horizontal line after each robot type
        latex_content.append(r"\hline")
    
    latex_content.append(r"\end{tabular}")
    latex_content.append(r"\vspace{0.1cm}")
    latex_content.append(r"\caption{Runtime (in seconds) and shield trigger (in percentage) statistics across different environments. Ours refers to our methods using the combination of TRPO-SRO, Shield, and ACP.}")
    latex_content.append(r"\label{tab:rq3_runtime}")
    latex_content.append(r"\end{table}")
    
    latex_content.append(r"\end{document}")
    
    # Save the LaTeX content to file
    os.makedirs("./plot_main_figures", exist_ok=True)
    output_path = f"./plot_main_figures/{output_file}"
    with open(output_path, "w") as f:
        f.write("\n".join(latex_content))
    print(f"LaTeX table saved to {output_path}")
    
    # Also save just the table part for easy copying
    table_start = latex_content.index(r"\begin{table}[t]")
    table_end = latex_content.index(r"\end{table}") + 1
    table_content = latex_content[table_start:table_end]
    
    table_path = f"./plot_main_figures/table_only_{output_file}"
    with open(table_path, "w") as f:
        f.write("\n".join(table_content))
    print(f"Table-only content saved to {table_path}")
    
    # Print a simplified version to console
    print("\nGenerated Table Data:")
    for row in latex_content:
        if "&" in row and not r"\hline" in row:
            print(row)


def main():
    """Main function to run the processing and generate tables."""
    args = parse_args()

    print(f"Looking for evaluation results in: {args.base_folder}")
    print(f"Targeting algorithms: {', '.join(TARGET_ALGORITHMS)}")
    print("Targeting environments ending in '2'")
    print(f"Targeting metrics: {RUNTIME_METRIC}, {SHIELD_METRIC}")

    csv_files = find_csv_files(args.base_folder, TARGET_ALGORITHMS)

    if not csv_files:
        print("No relevant CSV files found for the specified criteria.")
        return

    print(f"Found {len(csv_files)} CSV files to process.")
    raw_data = process_metrics(csv_files)

    if raw_data.empty:
        print(f"No data found for metrics '{RUNTIME_METRIC}' or '{SHIELD_METRIC}' in the processed files.")
        return
    
    # Process the raw data to get stats
    metrics_dict = process_raw_data(raw_data)
    
    # Generate LaTeX tables
    generate_latex_table(metrics_dict, output_file=args.output_file)


if __name__ == "__main__":
    # Example command:
    # python plot_rq3.py --base_folder ./ood_evaluation_folder --output_file runtime_table.tex
    main()
