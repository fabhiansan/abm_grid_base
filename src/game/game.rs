use super::agent::{Agent, AgentType, DeadAgentsData};
use super::grid::{Grid, Terrain};
use crate::ShelterData;
use crate::SimulationData;
use rand::prelude::*; // Import Rng for random number generation
use rand::seq::SliceRandom;
use serde_json::json;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;

#[derive(Clone)]
pub struct Model {
    pub grid: Grid,
    pub agents: Vec<Agent>,
    pub dead_agents: usize,
    pub dead_agent_types: Vec<AgentType>,
}

impl Model {
    pub fn step(&mut self, step: u32, is_tsunami: bool, tsunami_number: usize) {
        let mut dead_agents_this_step = 0;

        if is_tsunami {
            println!("TSUNAMI IS COMING ----- {}", tsunami_number);

            // Check if tsunami_number is within bounds of tsunami_data
            if tsunami_number < self.grid.tsunami_data.len() {
                // Add debug info
                let total_positive_heights = self.grid.tsunami_data[tsunami_number]
                    .iter()
                    .map(|row| row.iter().filter(|&&height| height > 0).count())
                    .sum::<usize>();

                println!(
                    "Tsunami timestep {} has {} cells with positive height",
                    tsunami_number, total_positive_heights
                );

                // Print some tsunami height data points
                if total_positive_heights > 0 {
                    // Find and print first 5 positions with positive tsunami heights
                    let mut found_positions = 0;
                    for (y, row) in self.grid.tsunami_data[tsunami_number].iter().enumerate() {
                        for (x, &height) in row.iter().enumerate() {
                            if height > 0 {
                                println!("Tsunami at position ({}, {}): height {}", x, y, height);
                                found_positions += 1;
                                if found_positions >= 5 {
                                    break;
                                }
                            }
                        }
                        if found_positions >= 5 {
                            break;
                        }
                    }
                } else {
                    println!("WARNING: No positive tsunami heights in this timestep!");
                }

                // Print agent position stats
                let mut min_x = u32::MAX;
                let mut max_x = 0;
                let mut min_y = u32::MAX;
                let mut max_y = 0;

                for agent in &self.agents {
                    min_x = min_x.min(agent.x);
                    max_x = max_x.max(agent.x);
                    min_y = min_y.min(agent.y);
                    max_y = max_y.max(agent.y);
                }

                println!(
                    "Agent position range: x: ({} to {}), y: ({} to {})",
                    min_x, max_x, min_y, max_y
                );

                // Process agents in reverse order to safely remove them
                for i in (0..self.agents.len()).rev() {
                    let agent = &self.agents[i];

                    // Get tsunami height at agent position
                    let tsunami_height =
                        self.grid
                            .get_tsunami_height(tsunami_number, agent.x, agent.y);

                    // Debug print for first 5 agents
                    if i < 5 || tsunami_height > 0 {
                        println!(
                            "Agent {} at ({}, {}) - tsunami height: {}",
                            i, agent.x, agent.y, tsunami_height
                        );
                    }

                    // If tsunami height > 0, agent dies
                    if tsunami_height > 0 {
                        dead_agents_this_step += 1;
                        self.grid.remove_agent(agent.x, agent.y, i);
                        println!(
                            "Agent {} mati akibat tsunami pada koordinat ({}, {}) dengan ketinggian air {}",
                            i, agent.x, agent.y, tsunami_height
                        );

                        self.dead_agent_types.push(agent.agent_type);
                        self.agents.remove(i);
                    }
                }

                println!("Jumlah agen mati pada step ini: {}", dead_agents_this_step);
            } else {
                println!(
                    "Warning: No tsunami data found for tsunami number: {}",
                    tsunami_number
                );
            }
        }

        self.dead_agents += dead_agents_this_step;

        let mut rng = rand::thread_rng();

        // --- Trigger and Decision Logic (Paper 2 & 3) ---
        if is_tsunami {
            for agent in &mut self.agents {
                // Trigger: Mark when agent becomes aware (if not already)
                if agent.evacuation_trigger_time.is_none() {
                    agent.evacuation_trigger_time = Some(step);
                    // println!("Agent {} triggered at step {}", agent.id, step); // Optional debug
                }

                // Decision: Only decide once after being triggered
                if agent.evacuation_trigger_time.is_some() && !agent.has_decided_to_evacuate {
                    // Simple probability based on knowledge and household size
                    // Higher knowledge increases chance, larger household slightly decreases (placeholder logic)
                    let knowledge_factor = agent.knowledge_level as f32 / 100.0;
                    let household_factor = 1.0 - ((agent.household_size.saturating_sub(1)) as f32 * 0.05); // Small penalty per extra member
                    let evacuation_probability = (knowledge_factor * household_factor).clamp(0.0, 1.0);

                    if rng.gen::<f32>() < evacuation_probability {
                        agent.has_decided_to_evacuate = true;
                        // println!("Agent {} decided to evacuate (Prob: {:.2})", agent.id, evacuation_probability); // Optional debug
                    } else {
                         // println!("Agent {} decided NOT to evacuate (Prob: {:.2})", agent.id, evacuation_probability); // Optional debug
                    }
                }
            }
        }
        // --- End Trigger and Decision Logic ---


        let mut agent_order: Vec<usize> = (0..self.agents.len()).collect();

        // Reset remaining steps for all agents at the start of the main step
        for agent in &mut self.agents {
            agent.remaining_steps = agent.speed;
        }

        // --- Movement Loop ---
        // Determine max steps needed in this cycle based on fastest agent who decided to move
        let max_steps_needed = self.agents.iter()
                                .filter(|a| a.has_decided_to_evacuate) // Only consider agents who decided
                                .map(|a| a.speed)
                                .max()
                                .unwrap_or(0); // If no one decided, 0 steps

        for _ in 0..max_steps_needed {
            agent_order.shuffle(&mut rng);
            let mut reserved_cells = HashSet::new();
            let mut moves = Vec::new();

            for &id in &agent_order {
                // Ensure agent still exists (might have reached shelter in a previous sub-step)
                if id >= self.agents.len() { continue; } // Skip if agent index is out of bounds

                let agent = &self.agents[id];

                // --- Movement Modification ---
                // Only move if agent has decided to evacuate, has steps left, and is not already in shelter
                if agent.has_decided_to_evacuate && agent.remaining_steps > 0 && !self.is_in_shelter(agent.x, agent.y) {
                    if let Some((nx, ny, fallback)) = self.find_best_move(agent, &reserved_cells) {
                        reserved_cells.insert((nx, ny));
                        moves.push((id, nx, ny, fallback));
                    }
                }
                // --- End Movement Modification ---
            }


            // Define uphill speed penalty factor (adjust as needed)
            const UPHILL_SPEED_PENALTY_FACTOR: f64 = 0.5; // Reduced penalty: 0.5 extra steps per meter climbed

            // Apply moves
            for &(id, new_x, new_y, _fallback) in &moves { // Prefixed unused fallback
                 // Ensure agent still exists before applying move
                if id >= self.agents.len() { continue; }

                let (old_x, old_y) = {
                    // Need to re-borrow agent immutably here to get old coords
                    let agent = &self.agents[id];
                    (agent.x, agent.y)
                };

                self.grid.remove_agent(old_x, old_y, id);

                // Borrow agent mutably to update state
                let agent = &mut self.agents[id];
                let was_on_road = agent.is_on_road;
                agent.is_on_road =
                    self.grid.terrain[new_y as usize][new_x as usize] == Terrain::Road;

                if !was_on_road && agent.is_on_road {
                    // println!("Agent {} reached road at ({}, {})", id, new_x, new_y);
                }

                agent.x = new_x;
                agent.y = new_y;

                // Calculate step cost *before* checking shelter, as it applies regardless
                let mut step_cost = 1; // Base cost for any move
                if let Some(dtm_layer) = self.grid.environment_layers.get("dtm") {
                    let old_elevation = dtm_layer.get(old_y as usize)
                                             .and_then(|row| row.get(old_x as usize))
                                             .copied().unwrap_or(0.0);
                    let new_elevation = dtm_layer.get(new_y as usize)
                                             .and_then(|row| row.get(new_x as usize))
                                             .copied().unwrap_or(old_elevation);
                    let delta_z = new_elevation - old_elevation;
                    if delta_z > 0.0 { // Moving uphill
                        let uphill_penalty = (delta_z * UPHILL_SPEED_PENALTY_FACTOR).ceil().max(0.0) as u32;
                        step_cost += uphill_penalty;
                    }
                }

                // Deduct cost from remaining steps, ensuring it doesn't go below zero
                if agent.remaining_steps >= step_cost {
                    agent.remaining_steps -= step_cost;
                } else {
                    agent.remaining_steps = 0; // Agent used all remaining steps
                }
                // Optional: Log if a fallback move was taken
                // if fallback {
                //     println!("Agent {} took a fallback move to ({}, {})", id, new_x, new_y);
                // }

                // Check if the new position is a shelter
                let is_shelter_cell = self.is_in_shelter(new_x, new_y);
                let current_agent_count = self.agents.len(); // Get length before mutable borrows

                if is_shelter_cell {
                    // Agent reached shelter.
                    self.enter_shelter(id, new_x, new_y); // Call this first (borrows self mutably)

                    // Now borrow agent mutably *briefly* to update its state
                    if id < current_agent_count { // Use stored length
                        let agent = &mut self.agents[id];
                        // Deduct cost first, then set to 0 as they stopped
                        if agent.remaining_steps >= step_cost {
                            agent.remaining_steps -= step_cost;
                        } else {
                            agent.remaining_steps = 0;
                        }
                        agent.remaining_steps = 0; // Agent stops in shelter
                    }
                    // Remove agent from the grid cell representation (borrows self.grid mutably)
                    self.grid.remove_agent(new_x, new_y, id);
                } else {
                    // Agent moved to a non-shelter cell.
                    // Borrow agent mutably *briefly* to update remaining steps
                     if id < current_agent_count { // Use stored length
                        let agent = &mut self.agents[id];
                        if agent.remaining_steps >= step_cost {
                            agent.remaining_steps -= step_cost;
                        } else {
                            agent.remaining_steps = 0;
                        }
                    }
                    // Add agent to the new grid cell (borrows self.grid mutably)
                    self.grid.add_agent(new_x, new_y, id);
                }
            }
        }
    }

    pub fn is_in_shelter(&self, x: u32, y: u32) -> bool {
        // Cast grid dimensions to usize for comparison
        if y as usize >= self.grid.height as usize || x as usize >= self.grid.width as usize {
            return false; // Out of bounds
        }
        matches!(
            self.grid.terrain[y as usize][x as usize],
            Terrain::Shelter(_)
        )
    }

    pub fn enter_shelter(&mut self, agent_id: usize, x: u32, y: u32) {
        // Check agent index bounds first
        if agent_id >= self.agents.len() { return; }

        // Check grid coordinate bounds before accessing terrain
        let grid_height_usize = self.grid.height as usize;
        let grid_width_usize = self.grid.width as usize;
        let y_usize = y as usize;
        let x_usize = x as usize;

        if y_usize >= grid_height_usize || x_usize >= grid_width_usize {
             println!("Warning: Agent {} tried to enter shelter at out-of-bounds coords ({}, {})", agent_id, x, y);
             return; // Coordinates out of bounds
        }

        // Now safely access terrain and check if it's a shelter
        if let Terrain::Shelter(shelter_id) = self.grid.terrain[y_usize][x_usize] {
            // Borrow agent immutably *after* confirming shelter and bounds
            let agent = &self.agents[agent_id];
            self.grid.add_to_shelter(shelter_id, agent_id, agent.agent_type);
            // Optionally mark agent as safe or inactive here if needed
            // self.agents[agent_id].is_safe = true;
        }
        // If it's not a shelter cell, do nothing in this function
    }

    fn find_best_move(
        &self,
        agent: &Agent,
        reserved: &HashSet<(u32, u32)>,
    ) -> Option<(u32, u32, bool)> { // Returns: (next_x, next_y, is_fallback)
        let mut rng = rand::thread_rng();
        let dirs = [ (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1) ];
        const SQRT2: f64 = 1.41421356237;
        // Incentive for moving uphill (reduces cost) - adjust factor as needed
        const UPHILL_INCENTIVE_FACTOR: f64 = 1.0; // Moderate incentive

        let mut potential_moves: Vec<(f64, u32, u32)> = Vec::new(); // (total_estimated_cost, nx, ny)

        // Get current elevation
        let dtm_layer_opt = self.grid.environment_layers.get("dtm");
        let current_elevation = dtm_layer_opt.and_then(|dtm| {
            dtm.get(agent.y as usize).and_then(|row| row.get(agent.x as usize)).copied()
        }).unwrap_or(0.0);

        // Determine goal based on current terrain
        let is_on_road = self.grid.terrain[agent.y as usize][agent.x as usize] == Terrain::Road;
        let goal_distance_field = if is_on_road {
            &self.grid.distance_to_shelter
        } else {
            &self.grid.distance_to_road
        };

        // Evaluate neighbors
        for &(dx, dy) in &dirs {
            let nx_i = agent.x as i32 + dx;
            let ny_i = agent.y as i32 + dy;

            // Check bounds
            if nx_i < 0 || ny_i < 0 || nx_i >= self.grid.width as i32 || ny_i >= self.grid.height as i32 { continue; }
            let nx = nx_i as u32;
            let ny = ny_i as u32;
            let nx_usize = nx as usize;
            let ny_usize = ny as usize;

            // Check terrain and reservation
            let terrain_next = &self.grid.terrain[ny_usize][nx_usize];
            if *terrain_next == Terrain::Blocked || reserved.contains(&(nx, ny)) { continue; }

            // --- Calculate Costs ---
            // Base movement cost (distance)
            let delta_d = if dx == 0 || dy == 0 { self.grid.cellsize } else { self.grid.cellsize * SQRT2 };

            // Elevation effect (incentive for uphill)
            let elevation_effect = if let Some(dtm) = dtm_layer_opt {
                let next_elevation = dtm.get(ny_usize)
                                        .and_then(|row| row.get(nx_usize))
                                        .copied()
                                        .unwrap_or(current_elevation); // Use current if next is invalid
                let delta_z = next_elevation - current_elevation;
                // Negative cost contribution if moving uphill (incentive)
                -(delta_z.max(0.0) * UPHILL_INCENTIVE_FACTOR)
            } else {
                0.0 // No elevation data, no effect
            };

            // Step cost = distance + elevation_effect (lower cost for uphill moves)
            let step_cost = delta_d + elevation_effect;

            // Heuristic cost depends on whether the agent is on road AND whether the neighbor is road
            let heuristic_cost = if is_on_road {
                // Agent is ON road, always use distance_to_shelter
                self.grid.distance_to_shelter[ny_usize][nx_usize].unwrap_or(f64::MAX / 2.0)
            } else {
                // Agent is OFF road
                if *terrain_next == Terrain::Road {
                    // Neighbor is ROAD, use distance_to_shelter from that road point
                    self.grid.distance_to_shelter[ny_usize][nx_usize].unwrap_or(f64::MAX / 2.0)
                } else {
                    // Neighbor is NOT ROAD, use distance_to_road
                    self.grid.distance_to_road[ny_usize][nx_usize].unwrap_or(f64::MAX / 2.0)
                }
            };

            // Total estimated cost (A*-like)
            // Goal cell definition also needs refinement based on context
            let is_target_goal_cell = if is_on_road {
                 matches!(terrain_next, Terrain::Shelter(_)) // On road -> target is Shelter
            } else {
                 *terrain_next == Terrain::Road // Off road -> target is Road
            };
            // Use heuristic cost unless the neighbor IS the immediate target goal
            let total_estimated_cost = if is_target_goal_cell { step_cost } else { step_cost + heuristic_cost };


            // Check if terrain is valid for the current goal (Simplified: handled by heuristic choice)
            // We only push moves where heuristic is not MAX/2.0 (effectively checking reachability)
             if heuristic_cost < (f64::MAX / 3.0) { // Check if the heuristic indicates a potentially valid path
                 // Ensure terrain is not blocked (already checked above)
                 // Allow Land/Road if off-road, Road/Shelter if on-road
                 let valid_terrain_type = if is_on_road {
                     *terrain_next == Terrain::Road || matches!(terrain_next, Terrain::Shelter(_))
                 } else {
                     *terrain_next == Terrain::Land || *terrain_next == Terrain::Road
                 };

                 if valid_terrain_type {
                    potential_moves.push((total_estimated_cost, nx, ny));
                 }
             }
        }

        // --- Choose Best Move ---
        if !potential_moves.is_empty() {
            // Sort by the calculated cost (which now favors uphill)
            potential_moves.sort_by(|a, b| a.0.total_cmp(&b.0));
            let (_best_cost, best_nx, best_ny) = potential_moves[0];

            // Check if the best path seems viable (heuristic isn't excessively high)
            let best_heuristic = goal_distance_field[best_ny as usize][best_nx as usize].unwrap_or(f64::MAX);
            if best_heuristic < (f64::MAX / 3.0) { // Avoid division by zero or near-infinite heuristics
                // Return the move with the lowest cost (incentivizing uphill)
                return Some((best_nx, best_ny, false)); // false indicates not a fallback move
            }
            // If best heuristic is too high, fall through to fallback logic
        }

        // --- Fallback Logic (If no suitable cost-based move was found or best path seems blocked) ---
        // Re-initialize rng for fallback choice
        // Note: We re-use the existing 'rng' from the start of the function
        let mut valid_neighbors = Vec::new();

        // 1. Find all valid, non-blocked neighbors
        for &(dx, dy) in &dirs {
            let nx_i = agent.x as i32 + dx;
            let ny_i = agent.y as i32 + dy;
            if nx_i >= 0 && ny_i >= 0 && nx_i < self.grid.width as i32 && ny_i < self.grid.height as i32 {
                let nx = nx_i as u32;
                let ny = ny_i as u32;
                if self.grid.terrain[ny as usize][nx as usize] != Terrain::Blocked {
                    valid_neighbors.push((nx, ny));
                }
            }
        }

        if valid_neighbors.is_empty() {
            return None; // No valid neighbors at all
        }

        // 2. Prioritize empty, non-reserved valid neighbors
        let empty_non_reserved: Vec<&(u32, u32)> = valid_neighbors
            .iter()
            .filter(|&&(nx, ny)| !reserved.contains(&(nx, ny)) && self.grid.agents_in_cell[ny as usize][nx as usize].is_empty())
            .collect();

        if !empty_non_reserved.is_empty() {
            if let Some(chosen) = empty_non_reserved.choose(&mut rng) {
                return Some((chosen.0, chosen.1, true)); // true indicates fallback move
            }
        }

        // 3. If no empty non-reserved, consider ANY non-reserved valid neighbor (might be occupied)
        let any_non_reserved: Vec<&(u32, u32)> = valid_neighbors
            .iter()
            .filter(|&&(nx, ny)| !reserved.contains(&(nx, ny)))
            .collect();

        if !any_non_reserved.is_empty() {
            if let Some(chosen) = any_non_reserved.choose(&mut rng) {
                return Some((chosen.0, chosen.1, true)); // true indicates fallback move
            }
        }

        // If we reach here, it means:
        // - No cost-based move was suitable OR the best path seemed blocked.
        // - No empty, non-reserved neighbors were found.
        // - No occupied, non-reserved neighbors were found.
        // Therefore, the only remaining possibility is that all valid neighbors are reserved.

        // 4. All valid neighbors are reserved in this sub-step, so the agent must wait.
        // println!("Agent {} waiting (all valid neighbors reserved)", agent.id); // Debug
        return None; // No move possible in this sub-step
    }

    pub fn save_shelter_data(
        &self,
        death_json_counter: &Vec<serde_json::Value>,
        shelter_json_counter: &Vec<serde_json::Value>,
    ) -> std::io::Result<()> {
        let filename = "output/shelter_data.json";

        let mut shelter_counts: HashMap<String, ShelterAgentCounts> = HashMap::new();

        for (&shelter_id, agents) in &self.grid.shelter_agents {
            let shelter_key = format!("shelter_{}", shelter_id);
            let counts = shelter_counts
                .entry(shelter_key)
                .or_insert_with(ShelterAgentCounts::new); // Use or_insert_with for efficiency

            for &(_, agent_type) in agents {
                match agent_type {
                    AgentType::Child => counts.child += 1,
                    AgentType::Teen => counts.teen += 1,
                    AgentType::Adult => counts.adult += 1,
                    AgentType::Elder => counts.elder += 1,
                }
            }
        }

        let current_shelter_data: HashMap<String, serde_json::Value> = shelter_counts
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    json!({
                        "child": v.child,
                        "teen": v.teen,
                        "adult": v.adult,
                        "elder": v.elder,
                    }),
                )
            })
            .collect();

        let data = json!({
            "death_json_counter": death_json_counter,
            "shelter_json_counter": shelter_json_counter,
            "shelter_agent_types": current_shelter_data,
        });

        let file = File::create(filename)?;
        serde_json::to_writer_pretty(file, &data)?;
        println!("Updated simulation data in {}", filename);

        Ok(())
    }
} // <-- Closing brace for impl Model

// Ensure ShelterAgentCounts is defined outside impl Model
#[derive(Default, Debug)] // Added Default and Debug for convenience
pub struct ShelterAgentCounts {
    pub child: u32,
    pub teen: u32,
    pub adult: u32,
    pub elder: u32,
}

impl ShelterAgentCounts {
    pub fn new() -> Self {
        ShelterAgentCounts {
            child: 0,
            teen: 0,
            adult: 0,
            elder: 0,
        }
    }

    // to_json might not be needed if using serde directly
    // pub fn to_json(&self) -> serde_json::Value {
    //     json!({
    //         "child": self.child,
    //         "teen": self.teen,
    //         "adult": self.adult,
    //         "elder": self.elder,
    //     })
    // }
}
