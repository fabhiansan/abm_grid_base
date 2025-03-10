#!/usr/bin/env python3

import requests
import json
import time
import argparse
import sys

class SimulationRunner:
    """Client for running ABM Grid Base Simulation until completion"""

    def __init__(self, base_url="http://localhost:8080", location="pacitan"):
        """Initialize the client with the base URL of the API server"""
        self.base_url = base_url
        self.location = location
        print(f"Starting simulation runner for location: {location}")

    def health_check(self):
        """Check if the API server is running"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

    def get_config(self):
        """Get the current simulation configuration"""
        response = requests.get(f"{self.base_url}/config")
        return response.json()

    def update_config(self, config):
        """Update the simulation configuration"""
        # First get current config
        current_config = self.get_config()
        if "config" in current_config:
            current_config = current_config["config"]
            
        # Then update only the specified fields
        for key, value in config.items():
            current_config[key] = value
            
        # Send the updated config
        response = requests.post(f"{self.base_url}/config", json=current_config)
        return response.json()

    def reset_simulation(self):
        """Reset the simulation"""
        print("Resetting any existing simulation...")
        response = requests.post(f"{self.base_url}/reset")
        return response.json()
        
    def init_simulation(self, timeout=300):
        """Initialize the simulation with a specified location"""
        print(f"Initializing simulation for location: {self.location}...")
        
        # First reset any existing simulation
        self.reset_simulation()
        
        # Update the location in the configuration
        config = {"location": self.location}
        update_response = self.update_config(config)
        print("Configuration updated.")
        
        # Initialize the simulation
        try:
            response = requests.post(f"{self.base_url}/init", timeout=timeout)
            result = response.json()
            print("Simulation initialized successfully.")
            return result
        except requests.exceptions.Timeout:
            print("\nWARNING: Initialization request timed out after", timeout, "seconds.")
            print("This doesn't mean the initialization failed - the server is likely still processing.")
            print("Large data files can take a long time to load.")
            return {"status": "timeout", "message": "Request timed out, but server may still be processing"}

    def run_batch(self, batch_size=10):
        """Run a batch of simulation steps"""
        try:
            response = requests.post(f"{self.base_url}/run/{batch_size}")
            return response.json()
        except requests.exceptions.Timeout:
            print("Request timed out, will retry with a smaller batch size.")
            return None

    def get_status(self):
        """Get the current simulation status"""
        response = requests.get(f"{self.base_url}/status")
        return response.json()

    def export_results(self):
        """Export the simulation results"""
        response = requests.get(f"{self.base_url}/export")
        return response.json()

    def run_until_complete(self, batch_size=10, max_batches=1000):
        """Run the simulation until it completes or reaches the maximum number of batches"""
        print(f"Running simulation until completion (batch size: {batch_size})...")
        total_steps = 0
        batches_run = 0
        
        while batches_run < max_batches:
            # Run a batch of steps
            result = self.run_batch(batch_size)
            
            if not result:
                # If we got None due to timeout, try with a smaller batch size
                batch_size = max(1, batch_size // 2)
                print(f"Reducing batch size to {batch_size}")
                continue
                
            # Get the steps executed in this batch
            steps_executed = result.get("steps_executed", 0)
            total_steps += steps_executed
            batches_run += 1
            
            # Check if the simulation is completed
            sim_state = result.get("simulation_state", {})
            is_completed = sim_state.get("is_completed", False)
            current_step = sim_state.get("current_step", 0)
            dead_agents = sim_state.get("dead_agents", 0)
            
            print(f"Batch {batches_run}: Steps executed: {steps_executed}, " 
                  f"Total steps: {total_steps}, Current step: {current_step}, "
                  f"Dead agents: {dead_agents}")
            
            if is_completed:
                print("\nSimulation completed!")
                break
                
            if steps_executed == 0:
                print("\nNo steps were executed in this batch, simulation may be stuck.")
                break
                
            # Small pause to prevent overloading the server
            time.sleep(0.5)
        
        if batches_run >= max_batches:
            print("\nReached maximum number of batches without completion.")
        
        # Get the final status
        status = self.get_status()
        print("\nFinal simulation status:")
        print(json.dumps(status, indent=2))
        
        return total_steps

def main():
    parser = argparse.ArgumentParser(description="Run ABM Grid Base Simulation until completion")
    parser.add_argument("--host", default="localhost", help="API server host")
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    parser.add_argument(
        "--location", 
        choices=["jembrana", "pacitan"], 
        default="pacitan",
        help="Location data to use (jembrana or pacitan)"
    )
    parser.add_argument(
        "--batch", 
        type=int, 
        default=10,
        help="Number of steps to run in each batch"
    )
    parser.add_argument(
        "--initialize", 
        action="store_true",
        help="Initialize the simulation before running"
    )
    args = parser.parse_args()
    
    # Create the simulation runner
    base_url = f"http://{args.host}:{args.port}"
    runner = SimulationRunner(base_url, args.location)
    
    # Check if the server is running
    try:
        health = runner.health_check()
        print("Server health check: ", health.get("status", "unknown"))
    except Exception as e:
        print(f"Error connecting to the API server at {base_url}: {e}")
        sys.exit(1)
    
    # Initialize if requested
    if args.initialize:
        init_result = runner.init_simulation()
        if init_result.get("status") == "error":
            print("Error initializing simulation:", init_result.get("message", "Unknown error"))
            sys.exit(1)
    
    # Run until complete
    total_steps = runner.run_until_complete(batch_size=args.batch)
    
    print(f"\nExecution complete. Total steps run: {total_steps}")
    
    # Export results
    print("\nExporting results...")
    export_result = runner.export_results()
    print("Export status:", export_result.get("status", "unknown"))

if __name__ == "__main__":
    main()
