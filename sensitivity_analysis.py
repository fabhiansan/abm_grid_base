#!/usr/bin/env python3
"""
Sensitivity Analysis Tool for ABM Grid Base Simulation

This script automates sensitivity analysis by:
1. Running multiple simulations with varying parameters
2. Collecting and analyzing the exported data
3. Visualizing parameter-outcome relationships
"""

import requests
import json
import os
from itertools import product
import argparse
import subprocess
import sys

# Try to import visualization libraries, with helpful error messages if missing
try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install it with 'pip install pandas'")
    raise

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required. Install it with 'pip install matplotlib'")
    raise

try:
    import seaborn as sns
except ImportError:
    print("Error: seaborn is required. Install it with 'pip install seaborn'")
    raise

try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm is required. Install it with 'pip install tqdm'")
    raise

# Configuration
RUST_API_BASE_URL = "http://localhost:8080"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensitivity_analysis_results")

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def check_api_health():
    """Check if the Rust API is running"""
    try:
        response = requests.get(f"{RUST_API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("Rust API is running and healthy.")
            return True
        else:
            print(f"Rust API returned status code {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("Rust API is not running. Please start the API server.")
        return False

def get_default_config():
    """Get the default configuration from the API"""
    response = requests.get(f"{RUST_API_BASE_URL}/api/config", timeout=10)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get default config. Status code: {response.status_code}")
        return None

def flatten_dict(d, parent_key='', sep='.'):
    """Recursively flatten nested dictionaries."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def run_simulation(config, run_id, base_config):
    """Run a single simulation with the given configuration"""
    try:
        # Convert parameter format if needed (handle nested parameters)
        processed_config = base_config.copy()
        
        # Update processed_config with current run's parameters from 'config'
        # This loop handles both flat and nested parameters defined in param_ranges
        for key, value in config.items():
            if '.' in key:
                # Handle nested parameters with dot notation (e.g., "knowledge_level_distribution.high")
                parts = key.split('.')
                current = processed_config
                for i, part in enumerate(parts):
                    if i == len(parts) - 1:
                        current[part] = value
                    else:
                        if part not in current or not isinstance(current[part], dict):
                            current[part] = {}
                        current = current[part]
            else:
                # Handle regular parameters
                processed_config[key] = value

        # Handle "level" parameters by setting corresponding min and max for the API
        # The original "level" parameter (e.g., "milling_time_value") remains in the 'config'
        # dictionary that is saved with the results, which is good for plotting.
        # These 'level' parameters are expected to be in 'processed_config' at this point
        # if they were part of the 'config' (params) for the current run.
        if "milling_time_value" in processed_config:
            level = processed_config.pop("milling_time_value") 
            processed_config["milling_time_min"] = level
            processed_config["milling_time_max"] = level
        
        if "knowledge_level_value" in processed_config:
            level = processed_config.pop("knowledge_level_value")
            processed_config["knowledge_level_min"] = level
            processed_config["knowledge_level_max"] = level

        if "household_size_value" in processed_config:
            level = processed_config.pop("household_size_value")
            processed_config["household_size_min"] = level
            processed_config["household_size_max"] = level
            
        # Reset the simulation
        reset_response = requests.post(f"{RUST_API_BASE_URL}/api/reset", timeout=10)
        if reset_response.status_code != 200:
            print(f"Failed to reset simulation. Status code: {reset_response.status_code}")
            return None
        
        # Update the configuration
        config_response = requests.post(f"{RUST_API_BASE_URL}/api/config", json=processed_config, timeout=10)
        if config_response.status_code != 200:
            print(f"Failed to update config. Status code: {config_response.status_code}")
            return None
        
        # Initialize the simulation
        init_response = requests.post(f"{RUST_API_BASE_URL}/api/init", timeout=10000)
        if init_response.status_code != 200:
            print(f"Failed to initialize simulation. Status code: {init_response.status_code}")
            return None
        
        # Run the simulation for the specified number of steps
        current_max_steps = processed_config.get('max_steps', 100)
        print(f"DEBUG: Starting simulation run {run_id} for {current_max_steps} steps...")
        run_response = requests.post(f"{RUST_API_BASE_URL}/api/run/{current_max_steps}", timeout=10000) # Increased timeout to 5 minutes
        if run_response.status_code != 200:
            print(f"Failed to run simulation. Status code: {run_response.status_code}")
            return None
    except requests.exceptions.Timeout:
        print(f"Timeout occurred during run {run_id}. The simulation may be taking too long.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error occurred during run {run_id}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error occurred during run {run_id}: {e}")
        return None
    
    # Export agent outcomes
    print(f"DEBUG: Exporting data for run {run_id}...")
    outcomes_response = requests.get(f"{RUST_API_BASE_URL}/api/export/agent_outcomes", timeout=10000) # Increased timeout
    geojson_response = requests.get(f"{RUST_API_BASE_URL}/api/export/geojson", timeout=10000) # Increased timeout
    
    if outcomes_response.status_code == 200 and geojson_response.status_code == 200:
        # Save outcomes to file with run_id in the name
        outcomes_api_response = outcomes_response.json()
        geojson_api_response = geojson_response.json()
        
        run_dir = os.path.join(OUTPUT_DIR, f"run_{run_id}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        # Process Agent Outcomes
        actual_outcomes_content = outcomes_api_response # Default to API response
        outcomes_file_path_from_api = outcomes_api_response.get("file_path")
        if outcomes_file_path_from_api and outcomes_api_response.get("status") == "ok":
            try:
                # Ensure the path from API is treated as relative to the script's CWD or an absolute path
                resolved_actual_outcomes_path = os.path.abspath(outcomes_file_path_from_api)
                if os.path.exists(resolved_actual_outcomes_path):
                    with open(resolved_actual_outcomes_path, "r", encoding="utf-8") as src_f:
                        loaded_content = json.load(src_f)
                    with open(os.path.join(run_dir, "agent_outcomes.json"), "w", encoding="utf-8") as dest_f:
                        json.dump(loaded_content, dest_f, indent=2)
                    actual_outcomes_content = loaded_content # Use actual content
                    print(f"Successfully saved actual outcomes from {resolved_actual_outcomes_path} to {os.path.join(run_dir, 'agent_outcomes.json')} for run {run_id}")
                else:
                    print(f"Error: Actual outcomes file not found at {resolved_actual_outcomes_path} for run {run_id}. API response: {outcomes_api_response}. Saving API response instead.")
                    with open(os.path.join(run_dir, "agent_outcomes.json"), "w", encoding="utf-8") as f:
                        json.dump(outcomes_api_response, f, indent=2)
            except Exception as e:
                print(f"Error reading/writing actual outcomes data for run {run_id} from {outcomes_file_path_from_api}: {e}. API response: {outcomes_api_response}. Saving API response instead.")
                with open(os.path.join(run_dir, "agent_outcomes.json"), "w", encoding="utf-8") as f:
                    json.dump(outcomes_api_response, f, indent=2)
        else:
            print(f"Warning: 'file_path' not found or status not 'ok' in outcomes_response for run {run_id}. Saving API response. Response: {outcomes_api_response}")
            with open(os.path.join(run_dir, "agent_outcomes.json"), "w", encoding="utf-8") as f:
                json.dump(outcomes_api_response, f, indent=2)

        # Process Agent GeoJSON
        actual_geojson_content = geojson_api_response # Default to API response
        geojson_file_path_from_api = geojson_api_response.get("file_path")
        if geojson_file_path_from_api and geojson_api_response.get("status") == "ok":
            try:
                resolved_actual_geojson_path = os.path.abspath(geojson_file_path_from_api)
                if os.path.exists(resolved_actual_geojson_path):
                    with open(resolved_actual_geojson_path, "r", encoding="utf-8") as src_f:
                        loaded_content = json.load(src_f)
                    with open(os.path.join(run_dir, "agent_geojson.json"), "w", encoding="utf-8") as dest_f:
                        json.dump(loaded_content, dest_f, indent=2)
                    actual_geojson_content = loaded_content # Use actual content
                    print(f"Successfully saved actual geojson from {resolved_actual_geojson_path} to {os.path.join(run_dir, 'agent_geojson.json')} for run {run_id}")
                else:
                    print(f"Error: Actual geojson file not found at {resolved_actual_geojson_path} for run {run_id}. API response: {geojson_api_response}. Saving API response instead.")
                    with open(os.path.join(run_dir, "agent_geojson.json"), "w", encoding="utf-8") as f:
                        json.dump(geojson_api_response, f, indent=2)
            except Exception as e:
                print(f"Error reading/writing actual geojson data for run {run_id} from {geojson_file_path_from_api}: {e}. API response: {geojson_api_response}. Saving API response instead.")
                with open(os.path.join(run_dir, "agent_geojson.json"), "w", encoding="utf-8") as f:
                    json.dump(geojson_api_response, f, indent=2)
        else:
            print(f"Warning: 'file_path' not found or status not 'ok' in geojson_response for run {run_id}. Saving API response. Response: {geojson_api_response}")
            with open(os.path.join(run_dir, "agent_geojson.json"), "w", encoding="utf-8") as f:
                json.dump(geojson_api_response, f, indent=2)
        
        return {
            "config": config,
            "outcomes": actual_outcomes_content, 
            "geojson": actual_geojson_content,
            "run_id": run_id
        }
    else:
        print(f"Failed to export data. Outcomes status: {outcomes_response.status_code}, GeoJSON status: {geojson_response.status_code}")
        return None

def extract_outcomes_metrics(simulation_results):
    """Extract key metrics from simulation outcomes"""
    metrics = []
    
    for result in simulation_results:
        if not result:
            continue
            
        config = result["config"]
        # 'outcomes' from result is now expected to be the actual list of agent outcomes,
        # or the API response dictionary if reading the actual file failed.
        raw_data_for_outcomes = result["outcomes"] 
        
        agent_data = [] # Initialize as empty list
        
        if isinstance(raw_data_for_outcomes, list):
            # This is the expected case: raw_data_for_outcomes is the list of agent dicts
            agent_data = raw_data_for_outcomes
        elif isinstance(raw_data_for_outcomes, dict):
            # This is the fallback case: raw_data_for_outcomes is the API response dict
            if "file_path" in raw_data_for_outcomes: # Indicates it's likely the API response
                 print(f"Run {result.get('run_id', 'N/A')}: Metrics based on API response fallback, actual agent data not processed from file.")
            # agent_data remains [], which is consistent with .get("agent_outcomes", []) on an API response.
        else:
            print(f"Run {result.get('run_id', 'N/A')}: Unexpected data type for outcomes: {type(raw_data_for_outcomes)}. Agent data will be empty.")

        # Calculate aggregate metrics
        num_agents = len(agent_data)
        evacuated = sum(1 for agent in agent_data if agent.get("is_in_shelter", False))
        evacuation_rate = evacuated / num_agents if num_agents > 0 else 0
        
        avg_evacuation_time = 0
        evacuation_times = []
        
        for agent in agent_data:
            if agent.get("evacuation_time") is not None:
                evacuation_times.append(agent.get("evacuation_time"))
                
        if evacuation_times:
            avg_evacuation_time = sum(evacuation_times) / len(evacuation_times)
        
        # Create a record for this simulation run
        record = {
            "run_id": result["run_id"],
            "num_agents": num_agents,
            "evacuation_rate": evacuation_rate,
            "avg_evacuation_time": avg_evacuation_time,
            "max_evacuation_time": max(evacuation_times) if evacuation_times else 0,
            "min_evacuation_time": min(evacuation_times) if evacuation_times else 0,
        }
        
        # Add all configuration parameters, flatten nested dicts
        flat_config = flatten_dict(config)
        for key, value in flat_config.items():
            if isinstance(value, (int, float, bool)) or isinstance(value, str):
                record[f"param_{key}"] = value
        
        metrics.append(record)
    
    return pd.DataFrame(metrics)

def generate_parameter_combinations(param_ranges):
    """Generate all combinations of parameters for sensitivity analysis"""
    # Extract parameter names and their values
    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] for name in param_names]
    
    # Generate all combinations
    combinations = list(product(*param_values))
    
    # Convert to list of dictionaries
    configs = []
    for values in combinations:
        config = {name: value for name, value in zip(param_names, values)}
        configs.append(config)
    
    return configs

def plot_sensitivity_analysis(df, output_dir):
    """Create visualizations for sensitivity analysis"""
    print("DEBUG: Entered plot_sensitivity_analysis function.")
    if df is None or df.empty:
        print("DEBUG: DataFrame is None or empty in plot_sensitivity_analysis. Skipping plotting.")
        return

    sns.set_style('whitegrid')
    sns.set_context('talk', font_scale=1.1)
    
    # Identify parameter columns (they start with 'param_')
    param_cols = [col for col in df.columns if col.startswith('param_')]
    metric_cols = ['evacuation_rate', 'avg_evacuation_time', 'max_evacuation_time']
    
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # 1. Correlation matrix - filter for numeric columns only
    numeric_param_cols = [col for col in param_cols if pd.api.types.is_numeric_dtype(df[col])]
    correlation_cols = numeric_param_cols + metric_cols
    
    print(f"Computing correlation matrix with these numeric columns: {correlation_cols}")
    
    # Skip correlation matrix if no numeric parameters
    if len(correlation_cols) > 1:  # Need at least 2 columns for correlation
        plt.figure(figsize=(12, 10), dpi=150)
        corr_matrix = df[correlation_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix: Parameters vs Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'), dpi=300)
        plt.close()
    
    # 2. Pairplots for each parameter against metrics
    for param in param_cols:
        param_name = param.replace('param_', '')
        
        # Skip non-numeric parameters for plots that require numeric data
        if not pd.api.types.is_numeric_dtype(df[param]):
            print(f"Skipping non-numeric parameter: {param_name}")
            continue
            
        for metric in metric_cols:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x=param, y=metric, alpha=0.6, edgecolor='w', s=50)
            
            # Add trendline
            sns.regplot(data=df, x=param, y=metric, scatter=False, color='red')
            
            plt.title(f'Impact of {param_name} on {metric}')
            plt.xlabel(param_name)
            plt.ylabel(metric)
            plt.tight_layout()
            param_file_name = param_name.replace('.', '_')
            plt.savefig(os.path.join(plots_dir, f'{param_file_name}_vs_{metric}.png'), dpi=300)
            plt.close()
    
    # 3. Boxplots for categorical parameters
    for param in param_cols:
        param_name = param.replace('param_', '')
        unique_values = df[param].nunique()
        
        # Only create boxplots if there are a reasonable number of categories
        # and if the parameter isn't a complex object (which would be represented as strings)
        if 2 <= unique_values <= 10 and (pd.api.types.is_numeric_dtype(df[param]) or df[param].dtype == 'bool' or df[param].dtype == 'category'):
            for metric in metric_cols:
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df, x=param, y=metric)
                plt.title(f'Distribution of {metric} by {param_name}')
                plt.xlabel(param_name)
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                plt.tight_layout()
                param_file_name = param_name.replace('.', '_')
                plt.savefig(os.path.join(plots_dir, f'boxplot_{param_file_name}_vs_{metric}.png'), dpi=300)
                plt.close()
    
    # 4. Summary statistics table
    summary_stats = df[metric_cols].describe()
    summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
    
    # 5. Save the full processed dataset
    df.to_csv(os.path.join(output_dir, 'sensitivity_analysis_results.csv'), index=False)
    
    print(f"Plots and data saved to {plots_dir}")

def run_sensitivity_analysis(param_ranges, base_config, num_runs=1):
    """
    Run sensitivity analysis with the provided parameter ranges
    
    param_ranges: Dictionary mapping parameter names to lists of values to test
    base_config: Base configuration to modify
    num_runs: Number of runs for each parameter combination (for statistical significance)
    """
    ensure_output_dir()
    
    # Generate all parameter combinations
    param_combinations = generate_parameter_combinations(param_ranges)
    print(f"Running sensitivity analysis with {len(param_combinations)} parameter combinations")
    
    # Store all simulation results
    all_results = []
    run_id = 0
    
    # Run simulations for each parameter combination
    for params in tqdm(param_combinations, desc="Running parameter combinations"):
        for _ in range(num_runs):
            # Create a new configuration by updating the base config with the parameters
            config = base_config.copy()
            config.update(params)
            
            # Run the simulation with this configuration
            result = run_simulation(config, run_id, base_config)
            if result:
                all_results.append(result)
            
            run_id += 1
    
    # Process and analyze results
    print(f"DEBUG: Simulation loop finished. Number of results in all_results: {len(all_results)}")
    if all_results:
        # Let's see a snippet of what's in all_results if it's not too long
        if len(all_results) < 5:
            print(f"DEBUG: all_results content (first few): {all_results[:5]}")
        else:
            print(f"DEBUG: First result in all_results: {all_results[0] if all_results else 'None'}")

        print("DEBUG: Attempting to extract metrics from simulation results...")
        metrics_df = extract_outcomes_metrics(all_results)
        print(f"DEBUG: Metrics DataFrame info after extraction:")
        if metrics_df is not None and not metrics_df.empty:
            metrics_df.info()
            print(f"DEBUG: metrics_df.head():\n{metrics_df.head()}")
        else:
            print("DEBUG: metrics_df is None or empty.")
        
        print("DEBUG: Attempting to create visualizations...")
        plot_sensitivity_analysis(metrics_df, OUTPUT_DIR)
        
        print(f"Sensitivity analysis complete! Results saved to {OUTPUT_DIR}")
        
        return metrics_df
    else:
        print("No valid simulation results were obtained.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run sensitivity analysis on ABM simulation")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per parameter combination")
    parser.add_argument("--simple", action="store_true", help="Run a simplified analysis with fewer parameter combinations")
    parser.add_argument("--test", action="store_true", help="Use sample data generated by generate_sample_data.py for the simulation runs")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.test:
        print("--- Test mode enabled: Generating and using sample data ---")
        try:
            generate_script_path = os.path.join(script_dir, "generate_sample_data.py")
            
            print(f"Running {generate_script_path} to generate sample data...")
            # Run generate_sample_data.py with default arguments
            # Using sys.executable to ensure the correct python interpreter
            completed_process = subprocess.run(
                [sys.executable, generate_script_path], 
                capture_output=True, text=True, check=True, cwd=script_dir
            )
            print("generate_sample_data.py output:")
            print(completed_process.stdout)
            if completed_process.stderr:
                print("generate_sample_data.py errors:")
                print(completed_process.stderr)
            print("Sample data generated successfully in ./data_sample/ directory relative to the script.")

        except subprocess.CalledProcessError as e:
            print(f"Error running generate_sample_data.py: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            print("Exiting due to failure in sample data generation.")
            return 
        except FileNotFoundError:
            print(f"Error: generate_sample_data.py not found at {generate_script_path}.")
            print("Please ensure generate_sample_data.py is in the same directory as sensitivity_analysis.py.")
            print("Exiting due to missing sample data generator.")
            return
    
    # Check if the API is running
    if not check_api_health():
        return
    
    # Get the default configuration to use as a base
    base_config = get_default_config()
    if not base_config:
        return

    if args.test:
        if base_config: # Ensure base_config was fetched successfully
            print("Updating base_config to use generated sample data paths...")
            # data_sample directory is created by generate_sample_data.py in script_dir
            sample_data_dir = os.path.join(script_dir, "data_sample")
            
            # These are assumed keys for the Rust API's configuration.
            # If the actual keys are different, these need to be adjusted.
            # Using absolute paths to be safe, assuming the Rust server might have a different CWD.
            base_config["grid_file_path"] = os.path.join(sample_data_dir, "sample_grid.asc")
            base_config["agent_file_path"] = os.path.join(sample_data_dir, "sample_agents.asc")
            base_config["dtm_file_path"] = os.path.join(sample_data_dir, "sample_dtm.asc")
            base_config["siren_config_path"] = os.path.join(sample_data_dir, "siren_config.json")
            base_config["tsunami_data_path"] = os.path.join(sample_data_dir, "tsunami_ascii_sample") # Directory path
            
            print(f"Updated base_config for test mode: {json.dumps(base_config, indent=2)}")
        else:
            # This case should ideally be caught by the 'if not base_config: return' above,
            # but as a safeguard if logic changes:
            print("Cannot update base_config for test mode as it failed to load initially.")
            return
    
    # Define parameter ranges to test specific "levels" for min/max parameters
    # using 2 values per parameter to reduce total runs, as requested by the user.
    param_ranges = {
        "milling_time_value": [0, 20],           # Low and High milling durations
        "siren_effectiveness": [0.6, 1.0],       # Low and High siren effectiveness
        "knowledge_level_value": [20, 80],       # Low and High knowledge levels
        "household_size_value": [1, 5]            # Low and High household sizes
    }
    
    # The args.simple flag will no longer change which parameters are tested,
    # as the user wants to focus only on the attributes specified above.
    # If args.simple is True, the number of combinations will still be smaller
    # if the value lists above were different for simple vs. comprehensive.
    # Currently, they are the same, meaning args.simple has no effect on param_ranges.

    # Run the sensitivity analysis
    results = run_sensitivity_analysis(param_ranges, base_config, num_runs=args.runs)
    
    if results is not None:
        # Print some summary statistics
        print("\nSummary of results:")
        print(results.describe())

if __name__ == "__main__":
    # Print instructions
    print("\nSensitivity Analysis Tool for ABM Grid Base Simulation")
    print("===================================================\n")
    print("This tool runs multiple simulations with varying parameters")
    print("and analyzes how they affect evacuation outcomes.\n")
    print("Tips:")
    print("- Use --simple flag for a quicker analysis with fewer parameters")
    print("- Use --runs N to run each parameter set N times for statistical significance")
    print("- Use --test to run the analysis using freshly generated sample data")
    print("- Results will be saved in the 'sensitivity_analysis_results' directory\n")
    
    # Run the analysis
    main()
