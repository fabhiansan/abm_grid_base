#!/usr/bin/env python3
import requests
import json
import time
import sys

class SimulationClient:
    """Client for interacting with the ABM Grid Base Simulation API"""

    def __init__(self, base_url="http://localhost:8042"):
        """Initialize the client with the base URL of the API server"""
        self.base_url = base_url

    def health_check(self):
        """Check if the API server is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except requests.exceptions.ConnectionError:
            print("Error: Cannot connect to API server. Make sure it's running.")
            sys.exit(1)

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

    def init_simulation(self, timeout=300):
        """Initialize the simulation"""
        print("Initializing simulation (this might take several minutes for large datasets)...")
        try:
            response = requests.post(f"{self.base_url}/init", timeout=timeout)
            return response.json()
        except requests.exceptions.Timeout:
            print("\nWARNING: Initialization request timed out after", timeout, "seconds.")
            print("This doesn't mean the initialization failed - the server is likely still processing.")
            return {"status": "timeout", "message": "Request timed out, but server may still be processing"}

    def run_step(self):
        """Run a single simulation step"""
        response = requests.post(f"{self.base_url}/step")
        return response.json()

    def run_steps(self, steps):
        """Run multiple simulation steps"""
        response = requests.post(f"{self.base_url}/run/{steps}")
        return response.json()

    def get_status(self):
        """Get the current simulation status"""
        response = requests.get(f"{self.base_url}/status")
        return response.json()

    def reset_simulation(self):
        """Reset the simulation"""
        response = requests.post(f"{self.base_url}/reset")
        return response.json()

def main():
    # Initialize client
    client = SimulationClient()
    
    # Check if server is running
    print("Checking if API server is running...")
    health = client.health_check()
    print(f"Server status: {health['status']}")
    
    # Reset any existing simulation
    print("\nResetting any existing simulation...")
    client.reset_simulation()
    
    # Configure for Pacitan
    print("\nConfiguring for Pacitan dataset...")
    config = {
        "location": "pacitan"
    }
    client.update_config(config)
    
    # Initialize simulation
    print("\nInitializing simulation with Pacitan data...")
    init_result = client.init_simulation()
    print(f"Initialization result: {init_result['status']}")
    
    # Check initial status
    status = client.get_status()
    print(f"\nInitial status:")
    print(f"  Agents count: {status['agents_count']}")
    print(f"  Dead agents: {status['dead_agents']}")
    print(f"  Current step: {status['simulation_state']['current_step']}")
    print(f"  Is tsunami: {status['simulation_state']['is_tsunami']}")
    
    # Run simulation until tsunami is triggered and we see dead agents
    print("\nRunning simulation until tsunami is triggered...")
    max_steps = 150  # More than our TSUNAMI_DELAY setting of 100
    step_batch = 10
    
    for i in range(max_steps // step_batch):
        # Run a batch of steps
        client.run_steps(step_batch)
        
        # Check status after batch
        status = client.get_status()
        current_step = status['simulation_state']['current_step']
        is_tsunami = status['simulation_state']['is_tsunami']
        dead_agents = status['dead_agents']
        
        print(f"Batch {i+1}: Steps executed: {step_batch}, Current step: {current_step}, "
              f"Is tsunami: {is_tsunami}, Dead agents: {dead_agents}")
        
        # Break if we have dead agents or completed simulation
        if dead_agents > 0 or status['simulation_state']['is_completed']:
            print("\nFound dead agents or simulation completed!")
            break
            
    # Final status report
    print("\nFinal simulation status:")
    status = client.get_status()
    print(f"  Total steps executed: {status['simulation_state']['current_step']}")
    print(f"  Is tsunami: {status['simulation_state']['is_tsunami']}")
    print(f"  Tsunami index: {status['simulation_state']['tsunami_index']}")
    print(f"  Dead agents: {status['dead_agents']}")
    print(f"  Is completed: {status['simulation_state']['is_completed']}")

if __name__ == "__main__":
    main()
