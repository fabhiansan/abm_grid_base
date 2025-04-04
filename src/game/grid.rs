use crate::game::agent::{Agent, AgentType};
use crate::game::State; // Assuming State is defined elsewhere, maybe in game::mod.rs?
use std::collections::{BinaryHeap, HashMap}; // Removed HashSet, VecDeque
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub enum Terrain {
    Land, // Represents traversable non-road/shelter terrain
    Blocked,
    Road,
    Shelter(u32), // Includes shelter ID
}

#[derive(Clone, Debug)] // Removed PartialEq as HashMap<String, Vec<Vec<f64>>> doesn't derive it easily
pub struct Grid {
    pub width: u32,
    pub height: u32,
    pub xllcorner: f64,
    pub yllcorner: f64,
    pub cellsize: f64,
    pub terrain: Vec<Vec<Terrain>>,
    pub shelters: Vec<(u32, u32, u32)>, // (x, y, shelter_id)
    pub agents_in_cell: Vec<Vec<Vec<usize>>>,
    // Changed distance fields to f64 for weighted costs
    pub distance_to_road: Vec<Vec<Option<f64>>>,
    pub distance_to_shelter: Vec<Vec<Option<f64>>>,
    pub shelter_agents: HashMap<u32, Vec<(usize, AgentType)>>,
    pub tsunami_data: Vec<Vec<Vec<u32>>>, // [timestep][y][x] -> height
    pub nrow: u32, // Redundant with height? Keep for consistency with ASC header names
    pub ncol: u32, // Redundant with width? Keep for consistency
    // Added environment_layers field
    pub environment_layers: HashMap<String, Vec<Vec<f64>>>,
}

impl Grid {
    // Updated Grid::new
    pub fn new(width: u32, height: u32, xllcorner: f64, yllcorner: f64, cellsize: f64, ncol: u32, nrow: u32) -> Self {
        Grid {
            width,
            height,
            xllcorner,
            yllcorner,
            cellsize,
            terrain: vec![vec![Terrain::Blocked; width as usize]; height as usize], // Initialize terrain here
            shelters: Vec::new(), // Initialize shelters
            agents_in_cell: vec![vec![Vec::new(); width as usize]; height as usize],
            distance_to_road: vec![vec![None; width as usize]; height as usize],
            distance_to_shelter: vec![vec![None; width as usize]; height as usize],
            shelter_agents: HashMap::new(),
            tsunami_data: Vec::new(),
            nrow, // Keep nrow
            ncol, // Keep ncol
            environment_layers: HashMap::new(), // Initialize empty env layers map
        }
    }

    pub fn remove_agent(&mut self, x: u32, y: u32, agent_id: usize) {
        let y_usize = y as usize;
        let x_usize = x as usize;
        if y_usize < self.agents_in_cell.len() && x_usize < self.agents_in_cell[y_usize].len() {
            if let Some(index) = self.agents_in_cell[y_usize][x_usize].iter().position(|&id| id == agent_id) {
                self.agents_in_cell[y_usize][x_usize].remove(index);
            }
        }
    }

    pub fn add_to_shelter(&mut self, shelter_id: u32, agent_id: usize, agent_type: AgentType) {
        self.shelter_agents
            .entry(shelter_id)
            .or_insert_with(Vec::new)
            .push((agent_id, agent_type));
    }

    pub fn add_agent(&mut self, x: u32, y: u32, agent_id: usize) {
         let y_usize = y as usize;
         let x_usize = x as usize;
         if y_usize < self.agents_in_cell.len() && x_usize < self.agents_in_cell[y_usize].len() {
            self.agents_in_cell[y_usize][x_usize].push(agent_id);
         } else {
             eprintln!("Warning: Attempted to add agent {} to invalid cell ({}, {})", agent_id, x, y);
         }
    }

    // --- Distance Calculation Functions ---

    // Updated compute_distance_to_shelters to accept use_dtm flag
    pub fn compute_distance_to_shelters(&mut self, use_dtm: bool) {
        let mut dist = vec![vec![None; self.width as usize]; self.height as usize];
        let mut heap = BinaryHeap::new();
        const SQRT2: f64 = 1.41421356237; // Precompute sqrt(2)

        // Initialize shelters with cost 0
        for &(x, y, _) in &self.shelters {
            let x = x as usize;
            let y = y as usize;
            if y < dist.len() && x < dist[y].len() { // Bounds check
                dist[y][x] = Some(0.0);
                heap.push(State { cost: 0.0, x: x as u32, y: y as u32 });
            }
        }

        let dtm = self.environment_layers.get("dtm"); // Get DTM layer if loaded

        while let Some(State { cost, x, y }) = heap.pop() {
            let x_usize = x as usize;
            let y_usize = y as usize;

            if let Some(current_cost) = dist[y_usize][x_usize] {
                 // Use total_cmp for robust f64 comparison
                if cost.total_cmp(&current_cost) == std::cmp::Ordering::Greater { continue; }
            }

            // Iterate through 8 neighbors
            for &(dx, dy) in &[ (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1) ] {
                let nx_i = x as i32 + dx;
                let ny_i = y as i32 + dy;

                if nx_i >= 0 && ny_i >= 0 && nx_i < self.width as i32 && ny_i < self.height as i32 {
                    let nx_usize = nx_i as usize;
                    let ny_usize = ny_i as usize;

                    // Skip blocked terrain
                    if self.terrain[ny_usize][nx_usize] == Terrain::Blocked { continue; }

                    // --- Calculate Move Cost based on Slope (conditionally) ---
                    let delta_d = if dx == 0 || dy == 0 { self.cellsize } else { self.cellsize * SQRT2 };

                    let move_cost = if use_dtm && dtm.is_some() { // Check flag AND DTM existence
                        let dtm_layer = dtm.unwrap(); // Safe to unwrap due to is_some() check
                        let current_elev = dtm_layer.get(y_usize).and_then(|row| row.get(x_usize)).copied().unwrap_or(0.0);
                        let next_elev = dtm_layer.get(ny_usize).and_then(|row| row.get(nx_usize)).copied().unwrap_or(0.0);
                        let delta_z = next_elev - current_elev;
                        let slope = if delta_d > 1e-6 { delta_z / delta_d } else { 0.0 };

                        // Slope Cost Factor: Exponential penalty for uphill, capped
                        let slope_factor = if slope > 0.0 {
                            let capped_exp_arg = (slope * 5.0).min(10.0);
                            1.0 + capped_exp_arg.exp() - 1.0
                        } else {
                            1.0
                        };
                        delta_d * slope_factor // Apply slope factor
                    } else {
                        delta_d // Simple distance if use_dtm is false or DTM not loaded
                    };
                    // --- End Cost Calculation ---

                    let next_total_cost = cost + move_cost;

                    if dist[ny_usize][nx_usize].is_none() || next_total_cost < dist[ny_usize][nx_usize].unwrap() {
                        dist[ny_usize][nx_usize] = Some(next_total_cost);
                        heap.push(State { cost: next_total_cost, x: nx_i as u32, y: ny_i as u32 });
                    }
                }
            }
        }
        self.distance_to_shelter = dist;
        if use_dtm && dtm.is_some() {
            println!("Computed distance to shelters (Least Cost Path using DTM slope).");
        } else {
            println!("Computed distance to shelters (Simple Distance Path).");
        }
    }

    // Updated compute_distance_to_road to accept use_dtm flag
    pub fn compute_distance_to_road(&mut self, use_dtm: bool) {
        let mut dist = vec![vec![None; self.width as usize]; self.height as usize];
        let mut heap = BinaryHeap::new();
        const SQRT2: f64 = 1.41421356237;

        // Initialize road cells with cost 0
        for y in 0..self.height as usize {
            for x in 0..self.width as usize {
                if self.terrain[y][x] == Terrain::Road {
                    dist[y][x] = Some(0.0);
                    heap.push(State { cost: 0.0, x: x as u32, y: y as u32 });
                }
            }
        }

        let dtm = self.environment_layers.get("dtm");

        while let Some(State { cost, x, y }) = heap.pop() {
             let x_usize = x as usize;
             let y_usize = y as usize;

            if let Some(current_cost) = dist[y_usize][x_usize] {
                // Use total_cmp for robust f64 comparison
                if cost.total_cmp(&current_cost) == std::cmp::Ordering::Greater { continue; }
            }

            // Iterate through 8 neighbors
            for &(dx, dy) in &[ (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1) ] {
                let nx_i = x as i32 + dx;
                let ny_i = y as i32 + dy;

                if nx_i >= 0 && ny_i >= 0 && nx_i < self.width as i32 && ny_i < self.height as i32 {
                    let nx_usize = nx_i as usize;
                    let ny_usize = ny_i as usize;

                    // Skip blocked terrain
                    if self.terrain[ny_usize][nx_usize] == Terrain::Blocked { continue; }

                    // --- Calculate Move Cost based on Slope (conditionally) ---
                    let delta_d = if dx == 0 || dy == 0 { self.cellsize } else { self.cellsize * SQRT2 };

                    let move_cost = if use_dtm && dtm.is_some() { // Check flag AND DTM existence
                        let dtm_layer = dtm.unwrap(); // Safe to unwrap due to is_some() check
                        let current_elev = dtm_layer.get(y_usize).and_then(|row| row.get(x_usize)).copied().unwrap_or(0.0);
                        let next_elev = dtm_layer.get(ny_usize).and_then(|row| row.get(nx_usize)).copied().unwrap_or(0.0);
                        let delta_z = next_elev - current_elev;
                        let slope = if delta_d > 1e-6 { delta_z / delta_d } else { 0.0 };

                        // Slope Cost Factor: Exponential penalty for uphill, capped
                        let slope_factor = if slope > 0.0 {
                            let capped_exp_arg = (slope * 5.0).min(10.0);
                            1.0 + capped_exp_arg.exp() - 1.0
                        } else {
                            1.0
                        };
                        delta_d * slope_factor // Apply slope factor
                    } else {
                        delta_d // Simple distance if use_dtm is false or DTM not loaded
                    };
                    // --- End Cost Calculation ---

                    let next_total_cost = cost + move_cost;

                    if dist[ny_usize][nx_usize].is_none() || next_total_cost < dist[ny_usize][nx_usize].unwrap() {
                        dist[ny_usize][nx_usize] = Some(next_total_cost);
                        heap.push(State { cost: next_total_cost, x: nx_i as u32, y: ny_i as u32 });
                    }
                }
            }
        }
        self.distance_to_road = dist;
        // Correct the print statement based on whether DTM was used
        if use_dtm && dtm.is_some() {
            println!("Computed distance to roads (Least Cost Path using DTM slope).");
        } else {
             println!("Computed distance to roads (Simple Distance Path).");
        }
    }

    // --- Other Grid Methods ---

    pub fn is_valid_coordinate(&self, x: u32, y: u32) -> bool {
        x < self.width && y < self.height
    }

    pub fn get_terrain(&self, x: u32, y: u32) -> Option<&Terrain> {
        if self.is_valid_coordinate(x, y) {
            Some(&self.terrain[y as usize][x as usize])
        } else {
            None
        }
    }

    pub fn get_tsunami_height(&self, tsunami_index: usize, x: u32, y: u32) -> u32 {
        self.tsunami_data.get(tsunami_index)
            .and_then(|frame| frame.get(y as usize))
            .and_then(|row| row.get(x as usize))
            .copied()
            .unwrap_or(0)
    }
}

// --- Loading Functions ---

// Function to load a generic float ASC layer (like DTM)
pub fn load_float_asc_layer(path_str: &str) -> io::Result<(Vec<Vec<f64>>, u32, u32, f64, f64, f64)> {
    println!("Loading float ASC layer from: {}", path_str);
    let path = Path::new(path_str);
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let parse_header = |line_opt: Option<io::Result<String>>, name: &str| -> io::Result<String> { // Removed mut
        line_opt.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("Missing {} line", name)))?
    };
    let parse_value = |line: &str, name: &str| -> io::Result<f64> { // Removed mut
        line.split_whitespace().nth(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("Cannot parse {}", name)))?
            .parse::<f64>()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Invalid {} value: {}", name, e)))
    };

    let ncols_line = parse_header(lines.next(), "ncols")?;
    let nrows_line = parse_header(lines.next(), "nrows")?;
    let xll_line = parse_header(lines.next(), "xllcorner")?;
    let yll_line = parse_header(lines.next(), "yllcorner")?;
    let cellsize_line = parse_header(lines.next(), "cellsize")?;
    let nodata_line = parse_header(lines.next(), "NODATA_value")?;

    let ncols = parse_value(&ncols_line, "ncols")? as u32;
    let nrows = parse_value(&nrows_line, "nrows")? as u32;
    let xllcorner = parse_value(&xll_line, "xllcorner")?;
    let yllcorner = parse_value(&yll_line, "yllcorner")?;
    let cellsize = parse_value(&cellsize_line, "cellsize")?;
    let nodata_value = parse_value(&nodata_line, "NODATA_value")?;

    println!("  Header: {}x{}, cellsize={}, nodata={}", ncols, nrows, cellsize, nodata_value);

    let mut data: Vec<Vec<f64>> = Vec::with_capacity(nrows as usize);

    for (r, line_result) in lines.enumerate() {
         if r >= nrows as usize { break; }
         let line = line_result?;
         let row: Vec<f64> = line.split_whitespace()
            .filter_map(|token| token.parse::<f64>().ok())
            .map(|val| {
                if (val - nodata_value).abs() < 1e-6 { 0.0 } else { val } // Use 0.0 for NODATA for now
            })
            .collect();

        if row.len() != ncols as usize {
             return Err(io::Error::new(io::ErrorKind::InvalidData, format!("Row {} has {} columns, expected {}", r, row.len(), ncols)));
        }
        data.push(row);
    }

     if data.len() != nrows as usize {
         return Err(io::Error::new(io::ErrorKind::InvalidData, format!("Found {} rows, expected {}", data.len(), nrows)));
     }

    println!("  Successfully loaded {}x{} float data.", ncols, nrows);
    Ok((data, ncols, nrows, xllcorner, yllcorner, cellsize))
}


// Updated load_grid_from_ascii
pub fn load_grid_from_ascii(path_str: &str) -> io::Result<(Grid, Vec<Agent>)> {
    println!("Loading base grid from ASCII: {}", path_str);
    let path = Path::new(path_str);
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let parse_header = |line_opt: Option<io::Result<String>>, name: &str| -> io::Result<String> { // Removed mut
        line_opt.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("Missing {} line", name)))?
    };
    let parse_u32 = |line: &str, name: &str| -> io::Result<u32> { // Removed mut
        line.split_whitespace().nth(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("Cannot parse {}", name)))?
            .parse::<u32>()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Invalid {} value: {}", name, e)))
    };
     let parse_f64 = |line: &str, name: &str| -> io::Result<f64> { // Removed mut
        line.split_whitespace().nth(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("Cannot parse {}", name)))?
            .parse::<f64>()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Invalid {} value: {}", name, e)))
    };

    let ncols_line = parse_header(lines.next(), "ncols")?;
    let nrows_line = parse_header(lines.next(), "nrows")?;
    let xll_line = parse_header(lines.next(), "xllcorner")?;
    let yll_line = parse_header(lines.next(), "yllcorner")?;
    let cellsize_line = parse_header(lines.next(), "cellsize")?;
    let _nodata_line = parse_header(lines.next(), "NODATA_value")?;

    let ncols = parse_u32(&ncols_line, "ncols")?;
    let nrows = parse_u32(&nrows_line, "nrows")?;
    let xllcorner = parse_f64(&xll_line, "xllcorner")?;
    let yllcorner = parse_f64(&yll_line, "yllcorner")?;
    let cellsize = parse_f64(&cellsize_line, "cellsize")?;

    println!("  Header: {}x{}, cellsize={}, xll={}, yll={}", ncols, nrows, cellsize, xllcorner, yllcorner);

    let mut grid = Grid::new(ncols, nrows, xllcorner, yllcorner, cellsize, ncols, nrows);
    // Removed agent_positions Vec - agents loaded separately

    for (y, line_result) in lines.enumerate() {
        if y >= nrows as usize { break; }
        let line = line_result?;
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() != ncols as usize {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("Row {} has {} columns, expected {}", y, tokens.len(), ncols)));
        }

        for (x, token) in tokens.iter().enumerate() {
            grid.terrain[y][x] = match *token {
                // Treat "0" or "0.0" as Land now, assuming NODATA_value isn't actually used for blocking
                // If you have a different value for truly blocked areas, add a case for it.
                "0" | "0.0" => Terrain::Land,
                "1" => Terrain::Road,
                token if token.starts_with("20") => {
                    if let Ok(shelter_id) = token[2..].parse::<u32>() {
                        grid.shelters.push((x as u32, y as u32, shelter_id));
                        Terrain::Shelter(shelter_id)
                    } else {
                        eprintln!("Warning: Invalid shelter ID format '{}' at ({}, {}). Treating as Blocked.", token, x, y);
                        Terrain::Blocked
                    }
                }
                // Ignore agent codes if present in terrain file
                "3" | "4" | "5" | "6" => {
                     eprintln!("Warning: Agent code '{}' found in terrain grid file at ({}, {}). Treating as Blocked.", token, x, y);
                     Terrain::Blocked
                }
                _ => {
                    eprintln!("Warning: Unknown terrain code '{}' at ({}, {}). Treating as Blocked.", token, x, y);
                    Terrain::Blocked
                }
            };
        }
    }

     if grid.terrain.len() != nrows as usize {
         return Err(io::Error::new(io::ErrorKind::InvalidData, format!("Found {} rows, expected {}", grid.terrain.len(), nrows)));
     }

    // Compute initial distances (always without DTM at this stage)
    grid.compute_distance_to_shelters(false);
    grid.compute_distance_to_road(false);

    println!("  Successfully loaded grid terrain and computed initial distances (without DTM).");

    // Return grid and an empty agent list
    Ok((grid, Vec::new()))
}
