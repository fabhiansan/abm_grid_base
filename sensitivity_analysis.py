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

def run_simulation(config, run_id, base_config):
    """Run a single simulation with the given configuration"""
    try:
        # Convert parameter format if needed (handle nested parameters)
        processed_config = base_config.copy()
        
        # Process flat parameters with dot notation for nested structure
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
        init_response = requests.post(f"{RUST_API_BASE_URL}/api/init", timeout=30)
        if init_response.status_code != 200:
            print(f"Failed to initialize simulation. Status code: {init_response.status_code}")
            return None
        
        # Run the simulation for the specified number of steps
        run_response = requests.post(f"{RUST_API_BASE_URL}/api/run/{processed_config.get('max_steps', 100)}", timeout=120)
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
    outcomes_response = requests.get(f"{RUST_API_BASE_URL}/api/export/agent_outcomes", timeout=30)
    geojson_response = requests.get(f"{RUST_API_BASE_URL}/api/export/geojson", timeout=30)
    
    if outcomes_response.status_code == 200 and geojson_response.status_code == 200:
        # Save outcomes to file with run_id in the name
        outcomes_data = outcomes_response.json()
        geojson_data = geojson_response.json()
        
        run_dir = os.path.join(OUTPUT_DIR, f"run_{run_id}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        
        with open(os.path.join(run_dir, "agent_outcomes.json"), "w", encoding="utf-8") as f:
            json.dump(outcomes_data, f, indent=2)
        
        with open(os.path.join(run_dir, "agent_geojson.json"), "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)
        
        return {
            "config": config,
            "outcomes": outcomes_data,
            "geojson": geojson_data,
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
        outcomes = result["outcomes"]
        
        # Extract agent outcomes if available
        agent_data = outcomes.get("agent_outcomes", [])
        
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
        
        # Add all configuration parameters, filtering complex/nested objects
        for key, value in config.items():
            if isinstance(value, (int, float, bool)):
                record[f"param_{key}"] = value
            elif isinstance(value, str):
                # For string values, we'll store them but they won't be used in numerical analysis
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
        plt.figure(figsize=(12, 10))
        corr_matrix = df[correlation_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix: Parameters vs Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'))
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
            sns.scatterplot(data=df, x=param, y=metric)
            
            # Add trendline
            sns.regplot(data=df, x=param, y=metric, scatter=False, color='red')
            
            plt.title(f'Impact of {param_name} on {metric}')
            plt.xlabel(param_name)
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{param_name}_vs_{metric}.png'))
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
                plt.savefig(os.path.join(plots_dir, f'boxplot_{param_name}_vs_{metric}.png'))
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
    if all_results:
        print("Extracting metrics from simulation results...")
        metrics_df = extract_outcomes_metrics(all_results)
        
        print("Creating visualizations...")
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
    args = parser.parse_args()
    
    # Check if the API is running
    if not check_api_health():
        return
    
    # Get the default configuration to use as a base
    base_config = get_default_config()
    if not base_config:
        return
    
    # Define parameter ranges to test based on evacuation scenario
    if args.simple:
        # Simple analysis with fewer combinations
        param_ranges = {
            "knowledge_level_distribution.high": [0.1, 0.3, 0.5],
            "warning_system_coverage": [0.6, 0.9],
            "social_influence_factor": [0.3, 0.7],
        }
    else:
        # More comprehensive analysis
        param_ranges = {
            # Knowledge level factors
            "knowledge_level_distribution.low": [0.2, 0.4, 0.6],
            "knowledge_level_distribution.high": [0.1, 0.3, 0.5],
            
            # Warning system factors
            "warning_system_coverage": [0.5, 0.7, 0.9],
            "warning_system_threshold": [0.05, 0.1, 0.2],
            
            # Social factors
            "social_influence_factor": [0.2, 0.4, 0.6],
            "trust_in_authorities": [0.3, 0.6, 0.9],
            
            # Simulation parameters
            "max_steps": [100, 200]
        }
    
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
    print("- Results will be saved in the 'sensitivity_analysis_results' directory\n")
    
    # Run the analysis
    main()
