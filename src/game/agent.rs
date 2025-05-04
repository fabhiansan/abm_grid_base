use rand::prelude::*;
use rand::distr::weighted::WeightedIndex;
use rand::thread_rng; // Correct import location
use std::fmt;
use serde::Serialize; // Import Serialize at the top level

#[derive(Debug, PartialEq, Copy, Clone, Serialize)] // Add Serialize derive
pub enum AgentType {
    Child,
    Teen,
    Adult,
    Elder,
}

// Implement Display for AgentType so we can use to_string()
impl fmt::Display for AgentType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AgentType::Child => write!(f, "Child"),
            AgentType::Teen => write!(f, "Teen"),
            AgentType::Adult => write!(f, "Adult"),
            AgentType::Elder => write!(f, "Elder"),
        }
    }
} // End impl fmt::Display

#[derive(Debug, Clone, Serialize)] // Agent struct already derives Serialize
pub struct Agent {
    pub id: usize,
    pub x: u32,
    pub y: u32,
    pub speed: u32,
    pub remaining_steps: u32,
    pub is_on_road: bool,
    pub agent_type: AgentType, // This field requires AgentType to be Serialize
    pub is_alive: bool,
    // --- Added based on Paper 3 ---
    pub knowledge_level: u8, // Represents disaster knowledge (0-100)
    pub household_size: u8, // Simplified household size
    pub has_decided_to_evacuate: bool, // Flag for evacuation decision
    // --- Added based on Paper 2 ---
    pub evacuation_trigger_time: Option<u32>, // Step when triggered
    pub is_in_shelter: bool, // Flag to indicate if agent is currently in a shelter cell
    // --- Added for Milling Time (Paper 4) ---
    pub milling_steps_remaining: u32, // Steps remaining for pre-evacuation delay/milling
    // --- Added for Siren Influence Tracking ---
    pub moved_by_siren: bool, // Flag to indicate if the agent's evacuation decision was overridden by the siren
}

pub const BASE_SPEED: f64 = 2.66;

impl Agent {
    // Add config parameter to the new function
    pub fn new(id: usize, x: u32, y: u32, agent_type: AgentType, is_on_road: bool, config: &crate::api::SimulationConfig) -> Self {
        let speed = match agent_type {
            AgentType::Child => 0.8 * BASE_SPEED,
            AgentType::Teen => 1.0 * BASE_SPEED,
            AgentType::Adult => 1.0 * BASE_SPEED,
            AgentType::Elder => 0.7 * BASE_SPEED,
        } as u32;

        Agent {
            id,
            x,
            y,
            speed,
            remaining_steps: speed,
            is_on_road,
            agent_type,
            is_alive: true,
            // Use config ranges for random generation
            knowledge_level: thread_rng().gen_range(config.knowledge_level_min..=config.knowledge_level_max),
            household_size: thread_rng().gen_range(config.household_size_min..=config.household_size_max),
            has_decided_to_evacuate: false,
            evacuation_trigger_time: None,
            is_in_shelter: false,
            milling_steps_remaining: 0,
            moved_by_siren: false, // Initialize to false
        }
    }
}

impl AgentType {
    pub fn random() -> Self {
        let weights = [6.21, 13.41, 59.10, 19.89]; // Distribusi bobot
        let mut rng = thread_rng();
        let dist = WeightedIndex::new(&weights).unwrap();
        match dist.sample(&mut rng) {
            0 => AgentType::Child,
            1 => AgentType::Teen,
            2 => AgentType::Adult,
            3 => AgentType::Elder,
            _ => AgentType::Adult, // Default case
        }
    }
}


use serde::Deserialize; // Removed duplicate Serialize import

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum TransportMode {
    Walk,
    Car,
}

impl Default for TransportMode {
    fn default() -> Self {
        TransportMode::Walk
    }
}

#[derive(Serialize, Deserialize)]
pub struct DeadAgentsData {
    pub step: u32,
    pub dead_agents: usize,
}

// --- Structs and Enums for Outcome Logging ---

#[derive(Serialize, Clone, Debug, PartialEq)]
pub enum AgentFinalStatus {
    Dead,
    SafeInShelter,
    // Could add other statuses later if needed, e.g., AliveOutsideShelter
}

#[derive(Serialize, Clone, Debug)]
pub struct AgentOutcome {
    pub agent_id: usize,
    pub agent_type: AgentType,
    pub initial_knowledge_level: u8, // Assuming we want the initial value
    pub initial_household_size: u8, // Assuming we want the initial value
    pub final_status: AgentFinalStatus,
    pub final_step: u32, // Step of death or reaching shelter, or simulation end step
    pub moved_by_siren: bool, // Flag to indicate if the agent's evacuation decision was overridden by the siren
}
// --- End Structs and Enums for Outcome Logging ---
