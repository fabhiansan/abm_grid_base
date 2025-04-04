use super::agent::{Agent, AgentType}; // Removed DeadAgentsData
use super::grid::{Grid, Terrain};
// Removed: use crate::ShelterData;
// Removed: use crate::SimulationData;
use rand::prelude::*; // Import Rng trait for gen_bool, etc.
use rand::thread_rng; // Import thread_rng function
use rand::seq::SliceRandom;
// Removed: use serde_json::json;
use std::collections::{HashSet}; // Removed HashMap
// Removed: use std::fs::File;

#[derive(Clone)]
pub struct Model {
    pub grid: Grid,
    pub agents: Vec<Agent>,
    pub dead_agents: usize,
    pub dead_agent_types: Vec<AgentType>,
}

impl Model {
    // Updated signature to include agent_reaction_delay and use clearer names
    pub fn step(&mut self, current_step: u32, agent_reaction_delay: u32, is_tsunami_active: bool, tsunami_data_index: usize) {
        let mut dead_agents_this_step = 0;

        // Apply tsunami effects if it's active and data exists for the current index
        if is_tsunami_active && tsunami_data_index < self.grid.tsunami_data.len() {
            // Update the grid's current tsunami height map for pathfinding or other checks if needed
            // self.grid.current_tsunami_height = self.grid.tsunami_data[tsunami_data_index].clone(); // Optional: if grid needs the full map

            println!("TSUNAMI ACTIVE - Step: {}, Tsunami Data Index: {}", current_step, tsunami_data_index);

            // Check if tsunami_data_index is within bounds of tsunami_data
            // (Redundant check, already done above, but kept for safety)
            if tsunami_data_index < self.grid.tsunami_data.len() {
                // Add debug info
                let total_positive_heights = self.grid.tsunami_data[tsunami_data_index]
                    .iter()
                    .map(|row| row.iter().filter(|&&height| height > 0).count())
                    .sum::<usize>();

                println!(
                    "Tsunami timestep {} has {} cells with positive height",
                    tsunami_data_index, total_positive_heights
                );

                // Print some tsunami height data points
                if total_positive_heights > 0 {
                    // Find and print first 5 positions with positive tsunami heights
                    let mut found_positions = 0;
                    // Prefix y with _ as it's not used directly in the loop body
                    for (_y, row) in self.grid.tsunami_data[tsunami_data_index].iter().enumerate() {
                        // Prefix x with _ as it's not used directly in the loop body
                        for (_x, &height) in row.iter().enumerate() {
                            if height > 0 {
                                // println!("Tsunami at position ({}, {}): height {}", _x, _y, height); // Example if needed
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

                // Process agents in reverse order to safely remove them
                for i in (0..self.agents.len()).rev() {
                     // Ensure agent index is still valid after potential removals
                     if i >= self.agents.len() { continue; }

                    let agent = &self.agents[i];

                     // Agent must be alive to be affected by tsunami
                     if !agent.is_alive { continue; }

                    // Get tsunami height at agent position using the correct index
                    let tsunami_height =
                        self.grid
                            .get_tsunami_height(tsunami_data_index, agent.x, agent.y);

                    // If tsunami height > 0, agent dies
                    if tsunami_height > 0 {
                        dead_agents_this_step += 1;
                        self.grid.remove_agent(agent.x, agent.y, agent.id); // Use agent.id
                        println!(
                            "Agent {} died due to tsunami at ({}, {}) with height {}",
                            agent.id, agent.x, agent.y, tsunami_height
                        );

                        self.dead_agent_types.push(agent.agent_type);
                        // Mark agent as dead instead of removing immediately to avoid index issues
                        self.agents[i].is_alive = false;
                    }
                }

                println!("Agents killed by tsunami this step: {}", dead_agents_this_step);
            }
            // No need for the 'else' here as the outer condition already checks index bounds
        } else if is_tsunami_active {
             println!("Warning: Tsunami is active but tsunami data index {} is out of bounds (max {}). No tsunami deaths applied.", tsunami_data_index, self.grid.tsunami_data.len());
        }
        // If !is_tsunami_active, no tsunami deaths occur.

        self.dead_agents += dead_agents_this_step; // Update total dead count

        let mut rng = thread_rng(); // Keep using imported thread_rng as it's common practice

        // --- Trigger and Decision Logic ---
        // Agents only consider reacting *after* the agent_reaction_delay
        if current_step >= agent_reaction_delay {
            for agent in &mut self.agents {
                 // Agent must be alive to react
                 if !agent.is_alive { continue; }

                // Trigger condition: Agent reacts *only* based on agent_reaction_delay,
                // simulating a warning or independent decision time.
                let should_consider_evacuating = current_step >= agent_reaction_delay;

                // Trigger: Mark when agent *could* start reacting (if not already triggered)
                // This time is now independent of the actual tsunami arrival.
                if should_consider_evacuating && agent.evacuation_trigger_time.is_none() {
                    agent.evacuation_trigger_time = Some(current_step); // Record the step they *could* react
                }

                // Decision: Decide once, only after the reaction delay has passed.
                if agent.evacuation_trigger_time.is_some() && !agent.has_decided_to_evacuate {
                    // The decision logic itself remains the same (based on probability)
                    let knowledge_factor = agent.knowledge_level as f32 / 100.0;
                    let household_factor = 1.0 - ((agent.household_size.saturating_sub(1)) as f32 * 0.05);
                    let evacuation_probability = (knowledge_factor * household_factor).clamp(0.0, 1.0);

                    // Use gen_bool for probability check - warning suggests random_bool, but gen_bool is correct for probability
                    if rng.gen_bool(evacuation_probability as f64) {
                        agent.has_decided_to_evacuate = true;
                    }
                }
            }
        }
        // --- End Trigger and Decision Logic ---


        let mut agent_order: Vec<usize> = (0..self.agents.len()).collect();

        // Reset remaining steps for all agents at the start of the main step
        for agent in &mut self.agents {
             if agent.is_alive { // Only reset for alive agents
                 agent.remaining_steps = agent.speed;
             }
        }

        // --- Movement Loop ---
        let max_steps_needed = self.agents.iter()
                                .filter(|a| a.is_alive && a.has_decided_to_evacuate)
                                .map(|a| a.speed)
                                .max()
                                .unwrap_or(0);

        for _ in 0..max_steps_needed {
            agent_order.shuffle(&mut rng);
            let mut reserved_cells = HashSet::new();
            let mut moves = Vec::new(); // Store moves as (agent_index, nx, ny, fallback)

            for &agent_index in &agent_order {
                if agent_index >= self.agents.len() { continue; }
                let agent = &self.agents[agent_index];

                if agent.is_alive && agent.has_decided_to_evacuate && agent.remaining_steps > 0 && !self.is_in_shelter(agent.x, agent.y) {
                    if let Some((nx, ny, fallback)) = self.find_best_move(agent, &reserved_cells) {
                        reserved_cells.insert((nx, ny));
                        moves.push((agent_index, nx, ny, fallback));
                    }
                }
            }

            const UPHILL_SPEED_PENALTY_FACTOR: f64 = 0.5;

            // Apply moves - Refactored (again) for borrow checker

            // --- Stage 1: Calculate move details and costs ---
            let mut move_updates = Vec::new();
            for &(agent_index, new_x, new_y, _fallback) in &moves {
                if agent_index >= self.agents.len() || !self.agents[agent_index].is_alive {
                    continue;
                }

                let (old_x, old_y, agent_id, initial_remaining_steps) = {
                    let agent = &self.agents[agent_index];
                    (agent.x, agent.y, agent.id, agent.remaining_steps)
                };

                let mut step_cost = 1;
                if let Some(dtm_layer) = self.grid.environment_layers.get("dtm") {
                    let old_elevation = dtm_layer.get(old_y as usize)
                                             .and_then(|row| row.get(old_x as usize))
                                             .copied().unwrap_or(0.0);
                    let new_elevation = dtm_layer.get(new_y as usize)
                                             .and_then(|row| row.get(new_x as usize))
                                             .copied().unwrap_or(old_elevation);
                    let delta_z = new_elevation - old_elevation;
                    if delta_z > 0.0 {
                        let uphill_penalty = (delta_z * UPHILL_SPEED_PENALTY_FACTOR).ceil().max(0.0) as u32;
                        step_cost += uphill_penalty;
                    }
                }

                let is_shelter_cell = self.is_in_shelter(new_x, new_y);

                move_updates.push((
                    agent_index,
                    old_x,
                    old_y,
                    new_x,
                    new_y,
                    agent_id,
                    step_cost,
                    is_shelter_cell,
                    initial_remaining_steps,
                ));
            }

            // --- Stage 2: Apply Agent State Updates (Mutable borrow of self.agents) ---
            for &(agent_index, _old_x, _old_y, new_x, new_y, _agent_id, step_cost, is_shelter_cell, initial_remaining_steps) in &move_updates {
                 if agent_index >= self.agents.len() { continue; }
                 let agent = &mut self.agents[agent_index];
                 if !agent.is_alive { continue; }

                 agent.x = new_x;
                 agent.y = new_y;
                 agent.is_on_road = self.grid.terrain[new_y as usize][new_x as usize] == Terrain::Road;

                 if initial_remaining_steps >= step_cost {
                     agent.remaining_steps = initial_remaining_steps - step_cost;
                 } else {
                     agent.remaining_steps = 0;
                 }

                 agent.is_in_shelter = is_shelter_cell;
                 if is_shelter_cell {
                     agent.remaining_steps = 0;
                 }
            }

            // --- Stage 3: Apply Grid Updates (Mutable borrow of self.grid / self) ---
            for &(_agent_index, old_x, old_y, new_x, new_y, agent_id, _step_cost, is_shelter_cell, _initial_remaining_steps) in &move_updates {
                 self.grid.remove_agent(old_x, old_y, agent_id);
                 if is_shelter_cell {
                     self.enter_shelter(agent_id, new_x, new_y);
                 } else {
                     self.grid.add_agent(new_x, new_y, agent_id);
                 }
            }

        } // End sub-step loop
    } // End step function

    pub fn is_in_shelter(&self, x: u32, y: u32) -> bool {
        if y as usize >= self.grid.height as usize || x as usize >= self.grid.width as usize {
            return false;
        }
        matches!(
            self.grid.terrain[y as usize][x as usize],
            Terrain::Shelter(_)
        )
    }

    pub fn enter_shelter(&mut self, agent_id: usize, x: u32, y: u32) {
        let agent_option = self.agents.iter_mut().find(|a| a.id == agent_id);
        if agent_option.is_none() {
             println!("Warning: Agent {} not found when trying to enter shelter.", agent_id);
             return;
        }
        let agent = agent_option.unwrap();

        let grid_height_usize = self.grid.height as usize;
        let grid_width_usize = self.grid.width as usize;
        let y_usize = y as usize;
        let x_usize = x as usize;

        if y_usize >= grid_height_usize || x_usize >= grid_width_usize {
             println!("Warning: Agent {} tried to enter shelter at out-of-bounds coords ({}, {})", agent_id, x, y);
             return;
        }

        if let Terrain::Shelter(shelter_id) = self.grid.terrain[y_usize][x_usize] {
            self.grid.add_to_shelter(shelter_id, agent.id, agent.agent_type);
            agent.is_in_shelter = true;
            agent.is_alive = true;
            agent.remaining_steps = 0;
            println!("Agent {} entered shelter {}", agent.id, shelter_id);
        }
    }


    fn find_best_move(
        &self,
        agent: &Agent,
        reserved: &HashSet<(u32, u32)>,
    ) -> Option<(u32, u32, bool)> {
        let mut rng = thread_rng(); // Keep using imported thread_rng
        let dirs = [ (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1) ];
        const SQRT2: f64 = 1.41421356237;
        const UPHILL_INCENTIVE_FACTOR: f64 = 1.0;

        let mut potential_moves: Vec<(f64, u32, u32)> = Vec::new();

        let dtm_layer_opt = self.grid.environment_layers.get("dtm");
        let current_elevation = dtm_layer_opt.and_then(|dtm| {
            dtm.get(agent.y as usize).and_then(|row| row.get(agent.x as usize)).copied()
        }).unwrap_or(0.0);

        let is_on_road = self.grid.terrain[agent.y as usize][agent.x as usize] == Terrain::Road;
        let goal_distance_field = if is_on_road {
            &self.grid.distance_to_shelter
        } else {
            &self.grid.distance_to_road
        };

        for &(dx, dy) in &dirs {
            let nx_i = agent.x as i32 + dx;
            let ny_i = agent.y as i32 + dy;

            if nx_i < 0 || ny_i < 0 || nx_i >= self.grid.width as i32 || ny_i >= self.grid.height as i32 { continue; }
            let nx = nx_i as u32;
            let ny = ny_i as u32;
            let nx_usize = nx as usize;
            let ny_usize = ny as usize;

            let terrain_next = &self.grid.terrain[ny_usize][nx_usize];
            if *terrain_next == Terrain::Blocked || reserved.contains(&(nx, ny)) { continue; }

            let delta_d = if dx == 0 || dy == 0 { self.grid.cellsize } else { self.grid.cellsize * SQRT2 };
            let elevation_effect = if let Some(dtm) = dtm_layer_opt {
                let next_elevation = dtm.get(ny_usize)
                                        .and_then(|row| row.get(nx_usize))
                                        .copied()
                                        .unwrap_or(current_elevation);
                let delta_z = next_elevation - current_elevation;
                -(delta_z.max(0.0) * UPHILL_INCENTIVE_FACTOR)
            } else {
                0.0
            };
            let step_cost = delta_d + elevation_effect;
            let heuristic_cost = goal_distance_field[ny_usize][nx_usize].unwrap_or(f64::MAX / 2.0);

            let is_target_goal_cell = if is_on_road {
                 matches!(terrain_next, Terrain::Shelter(_))
            } else {
                 *terrain_next == Terrain::Road
            };
            let total_estimated_cost = if is_target_goal_cell { step_cost } else { step_cost + heuristic_cost };

             if heuristic_cost < (f64::MAX / 3.0) {
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

        if !potential_moves.is_empty() {
            potential_moves.sort_by(|a, b| a.0.total_cmp(&b.0));
            let (_best_cost, best_nx, best_ny) = potential_moves[0];
            let best_heuristic = goal_distance_field[best_ny as usize][best_nx as usize].unwrap_or(f64::MAX);
            if best_heuristic < (f64::MAX / 3.0) {
                return Some((best_nx, best_ny, false));
            }
        }

        let mut valid_neighbors = Vec::new();
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
        if valid_neighbors.is_empty() { return None; }

        let empty_non_reserved: Vec<&(u32, u32)> = valid_neighbors
            .iter()
            .filter(|&&(nx, ny)| !reserved.contains(&(nx, ny)) && self.grid.agents_in_cell[ny as usize][nx as usize].is_empty())
            .collect();
        if !empty_non_reserved.is_empty() {
            if let Some(chosen) = empty_non_reserved.choose(&mut rng) {
                return Some((chosen.0, chosen.1, true));
            }
        }

        let any_non_reserved: Vec<&(u32, u32)> = valid_neighbors
            .iter()
            .filter(|&&(nx, ny)| !reserved.contains(&(nx, ny)))
            .collect();
        if !any_non_reserved.is_empty() {
            if let Some(chosen) = any_non_reserved.choose(&mut rng) {
                return Some((chosen.0, chosen.1, true));
            }
        }

        return None;
    }

/* // Method is currently unused by the API flow
    pub fn save_shelter_data(
        &self,
        death_json_counter: &Vec<serde_json::Value>,
        shelter_json_counter: &Vec<serde_json::Value>,
    ) -> std::io::Result<()> {
        let filename = "output/shelter_data.json";

        let mut final_shelter_counts: HashMap<String, ShelterAgentCounts> = HashMap::new();
        for (&shelter_id, agents_in_shelter) in &self.grid.shelter_agents {
            let shelter_key = format!("shelter_{}", shelter_id);
            let counts = final_shelter_counts.entry(shelter_key).or_default();
            for &(_, agent_type) in agents_in_shelter {
                match agent_type {
                    AgentType::Child => counts.child += 1,
                    AgentType::Teen => counts.teen += 1,
                    AgentType::Adult => counts.adult += 1,
                    AgentType::Elder => counts.elder += 1,
                }
            }
        }

        let final_shelter_data_json: HashMap<String, serde_json::Value> = final_shelter_counts
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
            "death_timeseries": death_json_counter,
            "shelter_timeseries": shelter_json_counter,
            "final_shelter_counts": final_shelter_data_json,
        });

        let file = File::create(filename)?;
        serde_json::to_writer_pretty(file, &data)?;
        println!("Updated simulation data in {}", filename);

        Ok(())
    }
*/
} // <-- Closing brace for impl Model

/* // Struct and impl are unused if save_shelter_data is unused
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
}
*/
