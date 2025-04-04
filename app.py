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
    """Fetches simulation status from the Rust API and returns metadata."""
    # This endpoint combines data from /status for the frontend
    return proxy_request('GET', '/status') # Simplified to use proxy helper

@app.route('/data/current_geojson')
def get_current_geojson():
    """Fetches the current agent locations as GeoJSON from the Rust API."""
    return proxy_request('GET', '/export/geojson', timeout=10) # Use proxy helper

@app.route('/init_simulation', methods=['POST'])
def init_simulation_proxy():
    """Proxies the POST request to the Rust API's /init endpoint."""
    return proxy_request('POST', '/init', timeout=20) # Use proxy helper, longer timeout

# --- New Proxy Endpoints ---

@app.route('/api/health', methods=['GET'])
def health_proxy():
    """Proxies GET /health"""
    return proxy_request('GET', '/health')

@app.route('/api/config', methods=['GET'])
def get_config_proxy():
    """Proxies GET /config"""
    return proxy_request('GET', '/config')

@app.route('/api/config', methods=['POST'])
def update_config_proxy():
    """Proxies POST /config"""
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "Missing JSON body for config update"}), 400
    return proxy_request('POST', '/config', json_data=json_data)

@app.route('/api/step', methods=['POST'])
def step_proxy():
    """Proxies POST /step"""
    return proxy_request('POST', '/step')

@app.route('/api/run/<int:steps>', methods=['POST'])
def run_steps_proxy(steps):
    """Proxies POST /run/{steps}"""
    # Add a longer timeout for potentially long runs
    return proxy_request('POST', f'/run/{steps}', timeout=60) 

@app.route('/api/export', methods=['GET'])
def export_proxy():
    """Proxies GET /export"""
    return proxy_request('GET', '/export')

@app.route('/api/reset', methods=['POST'])
def reset_proxy():
    """Proxies POST /reset"""
    return proxy_request('POST', '/reset')

@app.route('/api/grid/geojson', methods=['GET'])
def grid_geojson_proxy():
    """Proxies GET /grid/geojson"""
    return proxy_request('GET', '/grid/geojson', timeout=20) # Longer timeout potentially


@app.route('/api/tsunami/geojson', methods=['GET'])
def tsunami_geojson_proxy():
    """Proxies GET /tsunami/geojson"""
    # Tsunami data might change frequently, shorter timeout might be okay
    return proxy_request('GET', '/tsunami/geojson', timeout=5)

@app.route('/api/grid/costs', methods=['GET'])
def grid_costs_proxy():
    """Proxies GET /grid/costs"""
    # This might return a larger payload, increase timeout
    return proxy_request('GET', '/grid/costs', timeout=30)

# --- Main Execution ---
if __name__ == '__main__':
    # Make sure the requests library is installed: pip install requests
    logger.info("Starting Flask server for simulation visualization...")
    app.run(debug=True, host='0.0.0.0', port=5001)
