import requests
import json
import time
import argparse


class SimulationClient:
    """Client for interacting with the ABM Grid Base Simulation API"""

    def __init__(self, base_url="http://localhost:8080"):
        """Initialize the client with the base URL of the API server"""
        self.base_url = base_url

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

    def init_simulation(self, timeout=300):
        """Initialize the simulation
        
        Args:
            timeout: Request timeout in seconds (default: 300 seconds/5 minutes)
        """
        print("Initializing simulation (this might take several minutes for large datasets)...")
        try:
            response = requests.post(f"{self.base_url}/init", timeout=timeout)
            return response.json()
        except requests.exceptions.Timeout:
            print("\nWARNING: Initialization request timed out after", timeout, "seconds.")
            print("This doesn't mean the initialization failed - the server is likely still processing.")
            print("Large data files (especially for Jembrana) can take a long time to load.")
            print("You can check the server logs to see progress, or try again with a smaller dataset.")
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

    def export_results(self):
        """Export the simulation results"""
        response = requests.get(f"{self.base_url}/export")
        return response.json()
        
    def export_agent_geojson(self):
        """Export agent locations as GeoJSON"""
        response = requests.get(f"{self.base_url}/export/geojson")
        return response.json()

    def reset_simulation(self):
        """Reset the simulation"""
        response = requests.post(f"{self.base_url}/reset")
        return response.json()


def main():
    parser = argparse.ArgumentParser(description="Client for ABM Grid Base Simulation API")
    parser.add_argument("--host", default="localhost", help="API server host")
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    parser.add_argument(
        "--location",
        choices=["jembrana", "pacitan"],
        default="jembrana",
        help="Location data to use for simulation (jembrana or pacitan)"
    )
    parser.add_argument(
        "--action",
        choices=[
            "health",
            "config",
            "init", 
            "step", 
            "run", 
            "status", 
            "export", 
            "export-geojson",
            "reset",
            "demo"
        ],
        default="demo",
        help="Action to perform"
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    client = SimulationClient(base_url)

    if args.action == "health":
        response = client.health_check()
        print(json.dumps(response, indent=2))
    
    elif args.action == "config":
        response = client.get_config()
        print(json.dumps(response, indent=2))
    
    elif args.action == "init":
        # First update config with the selected location
        config = {
            "location": args.location
        }
        update_response = client.update_config(config)
        print("Updated configuration with location:", args.location)
        print(json.dumps(update_response, indent=2))
        
        # Then initialize the simulation
        response = client.init_simulation()
        print(json.dumps(response, indent=2))
    
    elif args.action == "step":
        response = client.run_step()
        print(json.dumps(response, indent=2))
    
    elif args.action == "run":
        response = client.run_steps(args.steps)
        print(json.dumps(response, indent=2))
    
    elif args.action == "status":
        response = client.get_status()
        print(json.dumps(response, indent=2))
    
    elif args.action == "export":
        response = client.export_results()
        print(json.dumps(response, indent=2))
    
    elif args.action == "export-geojson":
        response = client.export_agent_geojson()
        # Save GeoJSON to a file
        filename = "agent_locations.geojson"
        
        print(json.dumps(response, indent=2))
        
        with open(filename, "w") as f:
            json.dump(response, f, indent=2)
        print(f"GeoJSON data exported to {filename}")
    
    elif args.action == "reset":
        response = client.reset_simulation()
        print(json.dumps(response, indent=2))
    
    elif args.action == "demo":
        # Run a complete demo of the simulation
        print("Checking API server health...")
        response = client.health_check()
        print(json.dumps(response, indent=2))
        
        # Update location configuration
        print(f"\nSetting location to {args.location}...")
        config = {
            "location": args.location
        }
        update_response = client.update_config(config)
        print(json.dumps(update_response, indent=2))
        
        print("\nInitializing simulation...")
        response = client.init_simulation()
        print(json.dumps(response, indent=2))
        
        print("\nRunning simulation steps...")
        for i in range(5):  # Run 5 steps
            response = client.run_step()
            print(f"Step {i+1} result:")
            print(f"  Dead agents: {response['result']['dead_agents']}")
            print(f"  Current step: {response['simulation_state']['current_step']}")
            time.sleep(0.5)  # Pause between steps
        
        print("\nExporting results...")
        response = client.export_results()
        print("Results exported successfully")
        
        print("\nResetting simulation...")
        response = client.reset_simulation()
        print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
