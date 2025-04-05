#!/usr/bin/env python3
"""
Generate sample data for tsunami simulation testing.
This script creates:
1. A grid file ('sample_grid.asc') with terrain data (roads, shelters, blocked).
   Roads have varying widths (1 or 2 cells).
2. An agent file ('sample_agents.asc') with agent locations placed randomly (not on shelters).
3. A DTM file ('sample_dtm.asc') with simple elevation data.
4. Tsunami frames for simulation.

Accepts command-line arguments for grid dimensions. Shelters are placed adjacent to roads.
Example: python generate_sample_data.py --rows 50 --cols 50
"""

import os
import numpy as np
import argparse
import random # Import random for shuffling

# Ensure the output directory exists
DATA_DIR = "./data_sample"
os.makedirs(DATA_DIR, exist_ok=True)

# --- Grid Metadata (Defaults, can be overridden by args) ---
XLLCORNER = 500000.0  # UTM Zone 49S X coordinate
YLLCORNER = 9100000.0 # UTM Zone 49S Y coordinate
CELLSIZE = 10.0

def find_adjacent_non_road_cell(grid, road_coords, existing_shelters):
    """
    Tries to find a valid placement spot (non-road, non-shelter, in-bounds)
    adjacent (N, S, E, W) to a randomly chosen road cell.
    """
    nrows, ncols = grid.shape
    if not road_coords: # Handle case where no roads exist
        return None
    shuffled_road_coords = random.sample(road_coords, len(road_coords)) # Shuffle roads

    for r_road, c_road in shuffled_road_coords:
        # Check neighbors (N, S, E, W)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r_neighbor, c_neighbor = r_road + dr, c_road + dc

            # Check bounds
            if 0 <= r_neighbor < nrows and 0 <= c_neighbor < ncols:
                # Check if not road and not an existing shelter
                neighbor_val = grid[r_neighbor, c_neighbor]
                is_existing_shelter = (r_neighbor, c_neighbor) in existing_shelters
                # Ensure neighbor is not road (value 1)
                if neighbor_val != 1 and not is_existing_shelter:
                    print(f"  Found spot ({r_neighbor},{c_neighbor}) adjacent to road ({r_road},{c_road})")
                    return (r_neighbor, c_neighbor) # Found a valid spot

    print("  Warning: Could not find a valid adjacent non-road cell for a shelter.")
    return None # No suitable spot found

def create_environment_grid(nrows, ncols, output_path="./data_sample/sample_grid.asc"):
    """Create the environment grid file (roads, shelters, blocked)."""
    nodata_value_grid = 0 # NODATA for the environment grid is 0
    road_value = 1

    # Create a grid array filled with blocked terrain (0)
    terrain_grid = np.full((nrows, ncols), nodata_value_grid, dtype=int)

    # --- Place Roads with Varying Widths ---
    # Middle Horizontal Road (2 cells wide, if possible)
    mid_row = nrows // 2
    if mid_row < nrows: # Ensure mid_row is a valid index
        terrain_grid[mid_row, :] = road_value
        if mid_row + 1 < nrows: # Check bounds for second row
            terrain_grid[mid_row + 1, :] = road_value
            print(f"Generated 2-cell wide horizontal road at rows {mid_row}, {mid_row+1}")
        else:
            print(f"Generated 1-cell wide horizontal road at row {mid_row} (grid too small for 2)")
    else:
        print(f"Skipping middle horizontal road (grid too small: nrows={nrows})")


    # Middle Vertical Road (2 cells wide, if possible)
    mid_col = ncols // 2
    if mid_col < ncols: # Ensure mid_col is a valid index
        terrain_grid[:, mid_col] = road_value
        if mid_col + 1 < ncols: # Check bounds for second column
            terrain_grid[:, mid_col + 1] = road_value
            print(f"Generated 2-cell wide vertical road at columns {mid_col}, {mid_col+1}")
        else:
             print(f"Generated 1-cell wide vertical road at column {mid_col} (grid too small for 2)")
    else:
        print(f"Skipping middle vertical road (grid too small: ncols={ncols})")

    # Quarter Horizontal Road (1 cell wide)
    qtr_row = nrows // 4
    if qtr_row < nrows: # Ensure index is valid
        terrain_grid[qtr_row, :] = road_value
        print(f"Generated 1-cell wide horizontal road at row {qtr_row}")
    else:
        print(f"Skipping quarter horizontal road (grid too small: nrows={nrows})")


    # Quarter Vertical Road (1 cell wide)
    qtr_col = ncols // 4
    if qtr_col < ncols: # Ensure index is valid
        terrain_grid[:, qtr_col] = road_value
        print(f"Generated 1-cell wide vertical road at column {qtr_col}")
    else:
         print(f"Skipping quarter vertical road (grid too small: ncols={ncols})")
    # --- End Road Placement ---


    # Find all road cells (value 1) - still needed for shelter placement logic
    road_cells = np.where(terrain_grid == road_value)
    road_positions = list(zip(road_cells[0], road_cells[1])) # List of (row, col) tuples

    # --- Place Shelters ---
    placed_shelters_coords = [] # Store coords of placed shelters
    if not road_positions:
        print("Warning: No roads generated, cannot place shelters adjacent to roads.")
    else:
        shelter_ids = [201, 202]
        print("Placing shelters adjacent to roads:")
        for shelter_id in shelter_ids:
            shelter_pos = find_adjacent_non_road_cell(terrain_grid, road_positions, placed_shelters_coords)
            if shelter_pos:
                r, c = shelter_pos
                terrain_grid[r, c] = shelter_id
                placed_shelters_coords.append(shelter_pos) # Add to list of occupied shelter spots
                print(f"  Placed Shelter ID {shelter_id} at ({r},{c})")
            else:
                print(f"  Failed to place Shelter ID {shelter_id}")
    # --- End Shelter Placement ---


    # Write the environment grid to an ASCII file
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header using provided dimensions
        f.write(f"ncols {ncols}\n")
        f.write(f"nrows {nrows}\n")
        f.write(f"xllcorner {XLLCORNER}\n")
        f.write(f"yllcorner {YLLCORNER}\n")
        f.write(f"cellsize {CELLSIZE}\n")
        f.write(f"NODATA_value {nodata_value_grid}\n") # Use 0 for environment grid

        # Write grid data
        for row in terrain_grid:
            f.write(' '.join(map(str, row)) + '\n')

    print(f"Created environment grid ({nrows}x{ncols}) at {output_path}")
    # Return the final grid (needed for agent placement and DTM)
    return terrain_grid

# Modified function to accept the final environment grid
def create_agent_grid(environment_grid, output_path="./data_sample/sample_agents.asc"):
    """Create the agent grid file with agents placed randomly (not on shelters)."""
    nrows, ncols = environment_grid.shape
    nodata_value_agents = 0 # NODATA for the agent grid is 0

    # Create the agent grid, initialized with nodata_value
    agent_grid = np.full((nrows, ncols), nodata_value_agents, dtype=int)

    # --- Identify Valid Placement Spots ---
    # Find shelter locations (values >= 200)
    shelter_cells = np.where(environment_grid >= 200)
    shelter_positions = set(zip(shelter_cells[0], shelter_cells[1])) # Set for fast lookup

    # Create list of all possible coordinates
    all_coords = [(r, c) for r in range(nrows) for c in range(ncols)]

    # Filter out shelter locations
    valid_placement_coords = [coord for coord in all_coords if coord not in shelter_positions]
    # --- End Identification ---


    # Choose random positions for agents from valid spots
    np.random.seed(42)  # For reproducibility
    if not valid_placement_coords:
        print("Warning: No valid positions found to place agents (excluding shelters).")
        num_agents = 0
        agent_indices = []
    else:
        # Scale number of agents based on grid size (e.g., 1% of valid cells, capped)
        max_agents = 500 # Set a reasonable upper cap
        num_agents = min(max_agents, len(valid_placement_coords), int(len(valid_placement_coords) * 0.01) + 5) # At least 5, up to 1% or max_agents
        print(f"Attempting to place {num_agents} agents on {len(valid_placement_coords)} valid cells.")
        # Get indices of the chosen valid positions
        agent_indices = np.random.choice(len(valid_placement_coords), num_agents, replace=False)

    # Define agent types
    agent_types = {
        3: "Adult",
        4: "Child",
        5: "Teen",
        6: "Elder"
    }
    num_agent_types = len(agent_types)

    # Place agents onto the agent_grid at the chosen valid positions
    print("Placing agents:")
    placed_count = 0
    for i, valid_pos_idx in enumerate(agent_indices):
        row, col = valid_placement_coords[valid_pos_idx] # Get the actual (row, col)
        # Ensure placement is within bounds (should be, but safety check)
        if 0 <= row < nrows and 0 <= col < ncols:
             if agent_grid[row, col] == nodata_value_agents: # Check if cell is empty in agent grid
                agent_type = 3 + (i % num_agent_types)  # Cycle through agent types 3, 4, 5, 6
                agent_grid[row, col] = agent_type
                # print(f"  Placed {agent_types[agent_type]} agent (type {agent_type}) at position ({row}, {col})")
                placed_count += 1
             # else: # Should not happen with replace=False
             #    print(f"  Warning: Cell ({row}, {col}) already occupied.")
        else:
             print(f"  Warning: Skipping agent placement at invalid position ({row}, {col})")

    print(f"Placed {placed_count} agents.")

    # Write the agent_grid to an ASCII file
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header using provided dimensions
        f.write(f"ncols {ncols}\n")
        f.write(f"nrows {nrows}\n")
        f.write(f"xllcorner {XLLCORNER}\n")
        f.write(f"yllcorner {YLLCORNER}\n")
        f.write(f"cellsize {CELLSIZE}\n")
        f.write(f"NODATA_value {nodata_value_agents}\n") # Use 0 for agent grid

        # Write grid data from agent_grid
        for row_data in agent_grid:
            f.write(' '.join(map(str, row_data)) + '\n')

    print(f"Created agent grid ({nrows}x{ncols}) at {output_path}")
    return agent_grid # Return if needed, though not currently used elsewhere

# --- New DTM Generation Function ---
def create_dtm_grid(nrows, ncols, output_path="./data_sample/sample_dtm.asc"):
    """Creates a simple DTM grid file."""
    nodata_value_dtm = -9999.0 # Standard NODATA for float DTM

    # Create DTM with a base elevation, a slope, and some noise
    base_elevation = 5.0 # Keep base elevation
    dtm_grid = np.full((nrows, ncols), base_elevation, dtype=float)

    # --- Add a STEEPER Hill in the Top-Right Quadrant ---
    hill_peak_row = int(nrows * 0.25) # Peak around 1/4 down from top
    hill_peak_col = int(ncols * 0.75) # Peak around 3/4 from left
    hill_height = 50.0 # Increased hill height significantly
    hill_radius_factor = 0.15 # Decreased radius factor to make it steeper
    hill_radius_sq = (max(nrows, ncols) * hill_radius_factor) ** 2

    print(f"Adding STEEPER hill centered near ({hill_peak_row}, {hill_peak_col}) with max height {hill_height}")

    for r in range(nrows):
        for c in range(ncols):
            # Calculate squared distance from the peak
            dist_sq = float((r - hill_peak_row)**2 + (c - hill_peak_col)**2)
            # Gaussian-like decay based on distance squared
            # Avoid division by zero if radius is zero (e.g., 1x1 grid)
            if hill_radius_sq > 0:
                 elevation_gain = hill_height * np.exp(-dist_sq / (2 * hill_radius_sq))
                 dtm_grid[r, c] += elevation_gain
            elif r == hill_peak_row and c == hill_peak_col: # Handle 1x1 case
                 dtm_grid[r, c] += hill_height

    # --- End Hill ---

    # Add slightly more random noise everywhere
    noise = np.random.rand(nrows, ncols) * 2.0 - 1.0 # Noise between -1.0 and 1.0
    dtm_grid += noise

    # Ensure non-negative elevation (important if noise makes it negative)
    dtm_grid = np.maximum(dtm_grid, 0.0)

    # Example: Make roads slightly lower elevation (optional)
    # road_mask = (environment_grid == 1) # Assuming environment_grid is accessible or passed
    # dtm_grid[road_mask] = 9.5

    # Write the DTM grid to an ASCII file
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header consistent with other grids
        f.write(f"ncols {ncols}\n")
        f.write(f"nrows {nrows}\n")
        f.write(f"xllcorner {XLLCORNER}\n")
        f.write(f"yllcorner {YLLCORNER}\n")
        f.write(f"cellsize {CELLSIZE}\n")
        f.write(f"NODATA_value {nodata_value_dtm}\n")

        # Write grid data, formatting floats
        for row in dtm_grid:
            f.write(' '.join(map(lambda x: f"{x:.2f}", row)) + '\n') # Format to 2 decimal places

    print(f"Created DTM grid ({nrows}x{ncols}) at {output_path}")
    return dtm_grid
# --- End DTM Generation ---


def create_mock_tsunami_data(env_grid, num_frames=5, output_dir="./data_sample/tsunami_ascii_sample"):
    """Create mock tsunami data frames based on the environment grid dimensions."""
    # Use dimensions and metadata from the environment grid
    nrows_tsu, ncols_tsu = env_grid.shape
    nodata_value_tsu = -9999 # Tsunami NODATA is typically -9999

    # Clear existing tsunami files first
    if os.path.exists(output_dir):
        print(f"Clearing old tsunami files in {output_dir}...")
        for filename in os.listdir(output_dir):
            if filename.startswith("tsunami_") and filename.endswith(".asc"):
                try:
                    os.remove(os.path.join(output_dir, filename))
                except OSError as e:
                    print(f"Error removing old tsunami file {filename}: {e}")
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Create tsunami frames (simple wave example)
    for frame in range(num_frames):
        # Initialize tsunami grid with 0 height
        tsunami_grid = np.zeros((nrows_tsu, ncols_tsu), dtype=int)

        # Example: Simple wave advancing from left (column 0)
        wave_advancement = (frame + 1) / num_frames # Fraction of grid covered
        wave_front_col = int(wave_advancement * ncols_tsu)
        end_col = min(ncols_tsu, wave_front_col) # wave_front_col is the first column *not* inundated

        # Apply wave height from column 0 up to the wave front
        for col in range(0, end_col):
             for row in range(nrows_tsu):
                 # Example height logic: Higher near the 'coast' (e.g., row 0)
                 height = int(10 * (1 - (row / nrows_tsu))) # Height decreases away from row 0
                 tsunami_grid[row, col] = max(0, height) # Ensure non-negative height

        # Write the tsunami frame to an ASCII file
        output_path = os.path.join(output_dir, f"tsunami_{frame:03d}.asc")
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header consistent with environment grid
            f.write(f"ncols {ncols_tsu}\n")
            f.write(f"nrows {nrows_tsu}\n")
            f.write(f"xllcorner {XLLCORNER}\n")
            f.write(f"yllcorner {YLLCORNER}\n")
            f.write(f"cellsize {CELLSIZE}\n")
            f.write(f"NODATA_value {nodata_value_tsu}\n") # Use -9999 for tsunami

            # Write grid data
            for row in tsunami_grid:
                f.write(' '.join(map(str, row)) + '\n')

        print(f"Created tsunami frame {frame} at {output_path}")

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate sample data for tsunami simulation.")
    parser.add_argument("-r", "--rows", type=int, default=20, help="Number of rows for the grid.")
    parser.add_argument("-c", "--cols", type=int, default=20, help="Number of columns for the grid.")
    parser.add_argument("-f", "--frames", type=int, default=5, help="Number of tsunami frames to generate.")
    args = parser.parse_args()

    print(f"Starting sample data generation ({args.rows}x{args.cols}, {args.frames} frames)...")

    # 1. Generate environment grid (which also places shelters)
    environment_grid = create_environment_grid(args.rows, args.cols)

    # 2. Generate agent grid using the final environment grid to avoid shelters
    create_agent_grid(environment_grid) # Pass the grid directly

    # 3. Generate DTM grid
    create_dtm_grid(args.rows, args.cols) # Use args for dimensions

    # 4. Generate mock tsunami data based on the environment grid dimensions and args
    create_mock_tsunami_data(environment_grid, num_frames=args.frames)

    print("\nSample data generation complete!")
    # Provide clear paths to the generated files
    env_grid_path = os.path.join(DATA_DIR, "sample_grid.asc")
    agent_grid_path = os.path.join(DATA_DIR, "sample_agents.asc")
    dtm_grid_path = os.path.join(DATA_DIR, "sample_dtm.asc") # Added DTM path
    tsunami_dir_path = os.path.join(DATA_DIR, "tsunami_ascii_sample")
    print(f"Environment grid saved to: {env_grid_path}")
    print(f"Agent grid saved to: {agent_grid_path}")
    print(f"DTM grid saved to: {dtm_grid_path}") # Print DTM path
    print(f"Tsunami frames saved to: {tsunami_dir_path}/")
