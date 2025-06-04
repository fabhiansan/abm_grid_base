use super::agent::{Agent, AgentType}; // Removed DeadAgentsData
use super::grid::{Grid, Terrain};
// Removed: use crate::ShelterData;
// Removed: use crate::SimulationData;
use rand::prelude::*; // Import Rng trait for gen_bool, etc.
// use rand::thread_rng; // Import thread_rng function - No longer needed directly
use rand::seq::SliceRandom;
use rayon::prelude::*; // Import Rayon prelude
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
    // Updated signature to accept specific config values + original siren_config
    pub fn step(
        &mut self,
        current_step: u32,
        is_tsunami_active: bool,
        tsunami_data_index: usize,
        siren_config: Option<crate::api::SirenConfig>, // Back to Option<SirenConfig>
        milling_time_min: u32, // Added specific config param
        milling_time_max: u32, // Added specific config param
        siren_effectiveness_probability: f32, // Added specific config param
    ) {
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
                    for (_y, row) in self.grid.tsunami_data[tsunami_data_index].iter().enumerate() {
                        for (_x, &height) in row.iter().enumerate() {
                            if height > 0 {
                                found_positions += 1;
                                if found_positions >= 5 { break; }
                            }
                        }
                        if found_positions >= 5 { break; }
                    }
                } else {
                    println!("WARNING: No positive tsunami heights in this timestep!");
                }

                // Parallel identification of agents killed by tsunami
                let newly_dead_agent_data: Vec<(usize, u32, u32, AgentType)> = self.agents
                    .par_iter_mut() // Iterate mutably to mark is_alive
                    .filter(|agent| agent.is_alive)
                    .filter_map(|agent| {
                        let tsunami_height = self.grid.get_tsunami_height(tsunami_data_index, agent.x, agent.y);
                        if tsunami_height > 0 {
                            agent.is_alive = false; // Mark as dead
                            // println!( // This println in parallel can be messy, consider removing or conditionalizing
                            //     "Agent {} marked dead due to tsunami at ({}, {}) with height {}",
                            //     agent.id, agent.x, agent.y, tsunami_height
                            // );
                            Some((agent.id, agent.x, agent.y, agent.agent_type))
                        } else {
                            None
                        }
                    })
                    .collect();

                // Sequentially update grid and dead agent counts
                for (id, x, y, agent_type) in newly_dead_agent_data {
                    dead_agents_this_step += 1;
                    self.grid.remove_agent(x, y, id);
                    self.dead_agent_types.push(agent_type);
                }
                if dead_agents_this_step > 0 {
                    println!("Agents killed by tsunami this step: {}", dead_agents_this_step);
                }
            }
        } else if is_tsunami_active {
             println!("Warning: Tsunami is active but tsunami data index {} is out of bounds (max {}). No tsunami deaths applied.", tsunami_data_index, self.grid.tsunami_data.len());
        }
        self.dead_agents += dead_agents_this_step;

        // --- Trigger, Milling, and Decision Logic (agent_reaction_delay removed) ---
        // Parallelize agent decision-making
        self.agents.par_iter_mut().for_each(|agent| { // Changed to for_each, closure takes |agent|
            let mut rng = rand::thread_rng(); // RNG initialized per thread/task
            if !agent.is_alive { return; } // Skip dead agents (Rayon's for_each uses return like continue)

            // Trigger: Mark when agent *could* start reacting (if not already triggered)
            // This now happens potentially from step 0 (Reverted to original behavior)
            if agent.evacuation_trigger_time.is_none() {
                agent.evacuation_trigger_time = Some(current_step); // Record the step they *could* react

                // --- Assign Milling Time based on passed config range ---
                agent.milling_steps_remaining = rng.gen_range(milling_time_min..=milling_time_max);
                // println!("Agent {} triggered at step {}, assigned milling time: {} (Range: {}-{})", agent.id, current_step, agent.milling_steps_remaining, milling_time_min, milling_time_max); // Debug
            }

            // *** SIREN OVERRIDE LOGIC (Now comes first, can override milling) ***
            let mut is_agent_affected_by_siren = false;
            if !agent.has_decided_to_evacuate && agent.evacuation_trigger_time.is_some() {
                // Use the passed siren_config Option
                if let Some(siren_cfg) = &siren_config {
                    // Check if siren is active based on time
                    if current_step >= siren_cfg.activation_step {
                        // Calculate distance squared in grid cells (cheaper than sqrt)
                        let dx = agent.x as i64 - siren_cfg.x as i64; // Use siren_cfg
                        let dy = agent.y as i64 - siren_cfg.y as i64; // Use siren_cfg
                        let dist_sq = (dx * dx + dy * dy) as f64;
                        let radius_sq = (siren_cfg.radius_cells as f64) * (siren_cfg.radius_cells as f64); // Use siren_cfg

                        // Check if agent is within radius
                        if dist_sq <= radius_sq {
                            is_agent_affected_by_siren = true;
                        }
                    }
                }

                if is_agent_affected_by_siren {
                    // Check effectiveness probability using passed config value
                    if rng.gen_bool(siren_effectiveness_probability as f64) {
                        agent.has_decided_to_evacuate = true; // Override decision due to siren
                        agent.moved_by_siren = true; // Set flag as decision was overridden by siren
                        agent.milling_steps_remaining = 0; // Important: Reset milling time when siren is effective
                        // println!("Agent {} forced to evacuate by siren (Effective Prob: {:.2}) at step {}", agent.id, siren_effectiveness_probability, current_step); // Debug
                    } else {
                        // println!("Agent {} ignored siren (Ineffective Prob: {:.2}) at step {}", agent.id, 1.0 - siren_effectiveness_probability, current_step); // Debug
                    }
                }
            }
            // *** END SIREN OVERRIDE LOGIC ***

            // Milling Countdown: If triggered and milling, decrement counter.
            // Only continue milling if not overridden by siren
            if !agent.moved_by_siren && agent.milling_steps_remaining > 0 {
                agent.milling_steps_remaining -= 1;
                return; // Corrected from continue
            }

            // Decision: Decide once, only after milling time is over and if not already decided by siren
            if agent.evacuation_trigger_time.is_some()
               && agent.milling_steps_remaining == 0
               && !agent.has_decided_to_evacuate {
                // Logistic model from Paris flood study: P = 1/(1+exp(-(α0 + α1*K + α2*(H-1))))
                let k = agent.knowledge_level as f64;
                let h = agent.household_size as f64;
                let logit = -1.87 + 0.037 * k - 0.228 * (h - 1.0);
                let evacuation_probability = (1.0 / (1.0 + (-logit).exp())) as f32;
                
                if rng.gen_bool(evacuation_probability as f64) {
                    agent.has_decided_to_evacuate = true;
                } else {
                    // Agent decided NOT to evacuate initially
                    // println!("Agent {} decided NOT to evacuate initially (Prob: {:.2})", agent.id, evacuation_probability); // Debug
                }
            }
        });
        // --- End Trigger, Milling, and Decision Logic ---

        let mut rng = rand::thread_rng(); // Add RNG for the sequential movement loop

        let mut agent_order: Vec<usize> = (0..self.agents.len()).collect();

        // Reset remaining steps for all agents at the start of the main step
        for agent in &mut self.agents {
             if agent.is_alive { // Only reset for alive agents
                 agent.remaining_steps = agent.speed;
             }
        }

        // --- Movement Loop ---
        // Filter agents who are alive, have decided, AND finished milling
        let max_steps_needed = self.agents.iter()
                                .filter(|a| a.is_alive && a.has_decided_to_evacuate && a.milling_steps_remaining == 0)
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

                // --- Movement Modification ---
                // Only move if agent is alive, has decided, finished milling, has steps left, and not in shelter
                if agent.is_alive
                   && agent.has_decided_to_evacuate
                   && agent.milling_steps_remaining == 0 // Check milling is done
                   && agent.remaining_steps > 0
                   && !self.is_in_shelter(agent.x, agent.y)
                {
                    if let Some((nx, ny, fallback)) = self.find_best_move(agent, &reserved_cells) {
                        reserved_cells.insert((nx, ny));
                        moves.push((agent_index, nx, ny, fallback));
                    }
                }
                // --- End Movement Modification ---
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
        let mut rng = rand::thread_rng(); // Keep using imported thread_rng
        let dirs = [ (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1) ];
        const SQRT2: f64 = 1.41421356237;
        // Removed UPHILL_INCENTIVE_FACTOR as it was causing incoherence.
        // We will use UPHILL_SPEED_PENALTY_FACTOR from the step function's scope implicitly
        // by replicating the penalty calculation here.
        const HEURISTIC_WEIGHT: f64 = 10.0; // How strongly to prioritize reducing distance to goal
        const EXPLORATION_PROBABILITY: f64 = 0.1; // Chance to pick a non-optimal move to escape local optima

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
            // Calculate step_cost consistently with the actual movement cost in the step function
            let mut step_cost = delta_d; // Base cost is distance
            if let Some(dtm) = dtm_layer_opt {
                let next_elevation = dtm.get(ny_usize)
                                        .and_then(|row| row.get(nx_usize))
                                        .copied()
                                        .unwrap_or(current_elevation); // Use current if next is missing
                let delta_z = next_elevation - current_elevation;

                // Apply uphill penalty *consistently* with the main step loop's calculation
                if delta_z > 0.0 {
                    // Note: UPHILL_SPEED_PENALTY_FACTOR (0.5) is defined in the step function scope.
                    // We replicate the logic here. Ideally, this factor would be a shared constant.
                    let uphill_penalty = delta_z * 0.5; // Using 0.5 directly for now
                    step_cost += uphill_penalty;
                }
                // Optional: Add downhill benefit?
                // else if delta_z < 0.0 { step_cost *= 0.9; }
            }
            // step_cost now reflects physical distance + consistent uphill penalty
            let current_heuristic_cost = goal_distance_field[agent.y as usize][agent.x as usize].unwrap_or(f64::MAX / 2.0);
            let next_heuristic_cost = goal_distance_field[ny_usize][nx_usize].unwrap_or(f64::MAX / 2.0);
            // Calculate the change in heuristic cost. Negative means moving closer to the goal.
            let heuristic_delta = next_heuristic_cost - current_heuristic_cost;
            let is_target_goal_cell = if is_on_road {
                 matches!(terrain_next, Terrain::Shelter(_))
            } else {
                 *terrain_next == Terrain::Road
            };
            // New cost: physical step cost + weighted change in heuristic distance.
            // Prioritize moves that decrease the distance to the goal (negative delta).
            // The weight increases the influence of the heuristic compared to the step cost.
            // If the next cell *is* the goal, the cost is just the step cost.
            let total_estimated_cost = if is_target_goal_cell { step_cost } else { step_cost + heuristic_delta * HEURISTIC_WEIGHT };

             // Only consider moves where the next cell is reachable according to the distance field
             // (i.e., not effectively infinite distance).
             if next_heuristic_cost < (f64::MAX / 3.0) {
                 let valid_terrain_type = if is_on_road {
                     *terrain_next == Terrain::Road || matches!(terrain_next, Terrain::Shelter(_))
                 } else {
                     // Allow moving onto Land or Road when not currently on a road
                     *terrain_next == Terrain::Land || *terrain_next == Terrain::Road
                 };
                 if valid_terrain_type {
                    potential_moves.push((total_estimated_cost, nx, ny));
                 }
             }
        }

        if !potential_moves.is_empty() {
            potential_moves.sort_by(|a, b| a.0.total_cmp(&b.0));

            // --- Exploration Logic ---
            let mut chosen_move_index = 0; // Default to the best move
            if potential_moves.len() > 1 && rng.gen_bool(EXPLORATION_PROBABILITY) {
                // Choose a random *other* move from the potential list
                chosen_move_index = rng.gen_range(1..potential_moves.len());
                // println!("Agent {} exploring (chose move {} out of {})", agent.id, chosen_move_index + 1, potential_moves.len()); // Debug
            }
            // --- End Exploration Logic ---

            let (_chosen_cost, chosen_nx, chosen_ny) = potential_moves[chosen_move_index];

            // Check if the chosen move (best or explored) leads towards a reachable goal
            let chosen_next_heuristic = goal_distance_field[chosen_ny as usize][chosen_nx as usize].unwrap_or(f64::MAX);
            if chosen_next_heuristic < (f64::MAX / 3.0) {
                 // Ensure the chosen cell isn't reserved by another agent this sub-step
                 if !reserved.contains(&(chosen_nx, chosen_ny)) {
                    return Some((chosen_nx, chosen_ny, false)); // Return the chosen move
                 }
                 // If the chosen cell *is* reserved, fall through to fallback logic
            }
            // If the chosen move's heuristic is effectively infinite, or the cell was reserved,
            // it means no good path was found via heuristic/exploration. Fallback logic will handle this.
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
