from flask import Flask, render_template, jsonify, request # Added request
from flask_cors import CORS
import requests  # Import requests library
import logging
from collections import defaultdict
import time
import numpy as np # Keep numpy if needed for frontend compatibility checks, otherwise remove

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Base URL for the Rust simulation API
RUST_API_BASE_URL = "http://localhost:8080" # Default port for the Rust API

# --- Helper Function for Proxying ---
def proxy_request(method, path, **kwargs):
    """Helper function to forward requests to the Rust API."""
    url = f"{RUST_API_BASE_URL}{path}"
    try:
        logger.info(f"Proxying {method} request to: {url}")
        response = requests.request(method, url, timeout=kwargs.get('timeout', 15), json=kwargs.get('json_data'))
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        # Try to parse JSON, fall back to text if not possible
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            data = {"message": response.text} # Return raw text if not JSON

        logger.info(f"Received response {response.status_code} from {url}")
        return jsonify(data), response.status_code
        
    except requests.exceptions.RequestException as e:
        error_message = f"Failed to connect or communicate with simulation API at {url}: {str(e)}"
        logger.error(error_message)
        return jsonify({"error": error_message, "suggestion": "Ensure the Rust simulation server is running."}), 503 # Service Unavailable
    except Exception as e:
        error_message = f"An unexpected error occurred while proxying to {url}: {str(e)}"
        logger.error(error_message)
        return jsonify({"error": error_message}), 500

# --- Existing Endpoints ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/data/metadata')
def get_metadata():
    """Fetches simulation status from the Rust API (/api/status) and returns metadata."""
    # This endpoint combines data from /api/status for the frontend
    return proxy_request('GET', '/api/status') # Target path updated to include /api

@app.route('/data/current_geojson')
def get_current_geojson():
    """Fetches the current agent locations as GeoJSON from the Rust API (/api/export/geojson)."""
    return proxy_request('GET', '/api/export/geojson', timeout=10) # Target path updated

@app.route('/init_simulation', methods=['POST'])
def init_simulation_proxy():
    """Proxies the POST request to the Rust API's /api/init endpoint."""
    # Frontend calls /init_simulation, Rust expects /api/init
    # Increased timeout to 300 seconds (5 minutes) for potentially long initialization
    return proxy_request('POST', '/api/init', timeout=300) # Target path updated

# --- New Proxy Endpoints ---

@app.route('/api/health', methods=['GET'])
def health_proxy():
    """Proxies GET /health (Note: Rust health endpoint is at root)"""
    return proxy_request('GET', '/health') # Target path is correct

@app.route('/api/config', methods=['GET'])
def get_config_proxy():
    """Proxies GET /api/config"""
    return proxy_request('GET', '/api/config') # Add /api to target path

@app.route('/api/config', methods=['POST'])
def update_config_proxy():
    """Proxies POST /config"""
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "Missing JSON body for config update"}), 400
    return proxy_request('POST', '/api/config', json_data=json_data) # Add /api to target path

@app.route('/api/step', methods=['POST'])
def step_proxy():
    """Proxies POST /api/step"""
    return proxy_request('POST', '/api/step') # Add /api to target path

@app.route('/api/run/<int:steps>', methods=['POST'])
def run_steps_proxy(steps):
    """Proxies POST /api/run/{steps}"""
    # Add a longer timeout for potentially long runs
    return proxy_request('POST', f'/api/run/{steps}', timeout=60) # Add /api to target path

@app.route('/api/export', methods=['GET'])
def export_proxy():
    """Proxies GET /api/export"""
    return proxy_request('GET', '/api/export') # Add /api to target path

@app.route('/api/reset', methods=['POST'])
def reset_proxy():
    """Proxies POST /api/reset"""
    return proxy_request('POST', '/api/reset') # Add /api to target path

@app.route('/api/grid/geojson', methods=['GET'])
def grid_geojson_proxy():
    """Proxies GET /api/grid/geojson"""
    return proxy_request('GET', '/api/grid/geojson', timeout=20) # Add /api to target path


@app.route('/api/tsunami/geojson', methods=['GET'])
def tsunami_geojson_proxy():
    """Proxies GET /api/tsunami/geojson"""
    # Tsunami data might change frequently, shorter timeout might be okay
    return proxy_request('GET', '/api/tsunami/geojson', timeout=5) # Add /api to target path

@app.route('/api/grid/costs', methods=['GET'])
def grid_costs_proxy():
    """Proxies GET /api/grid/costs"""
    # This might return a larger payload, increase timeout
    return proxy_request('GET', '/api/grid/costs', timeout=30) # Add /api to target path

@app.route('/api/agent/<int:agent_id>', methods=['GET'])
def agent_info_proxy(agent_id):
    """Proxies GET /api/agent/{agent_id}"""
    # Frontend calls /api/agent/..., Rust expects /api/agent/...
    return proxy_request('GET', f'/api/agent/{agent_id}')

@app.route('/api/export/agent_outcomes', methods=['GET'])
def agent_outcomes_proxy():
    """Proxies GET /api/export/agent_outcomes"""
    return proxy_request('GET', '/api/export/agent_outcomes', timeout=15)

# --- Main Execution ---
if __name__ == '__main__':
    # Make sure the requests library is installed: pip install requests
    logger.info("Starting Flask server for simulation visualization...")
    app.run(debug=True, host='0.0.0.0', port=5001)
