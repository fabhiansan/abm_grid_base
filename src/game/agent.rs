use rand::prelude::*;
use rand::distr::weighted::WeightedIndex;
use rand::thread_rng; // Correct import location
use std::fmt;

#[derive(Debug, PartialEq, Copy, Clone)]
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
}

#[derive(Debug, Clone)]
pub struct Agent {
    pub id: usize,
    pub x: u32,
    pub y: u32,
    pub speed: u32,
    pub remaining_steps: u32,
    pub is_on_road: bool,
    pub agent_type: AgentType,
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
}

pub const BASE_SPEED: f64 = 2.66;

impl Agent {
    pub fn new(id: usize, x: u32, y: u32, agent_type: AgentType, is_on_road: bool, ) -> Self {
        let speed = match agent_type {
            AgentType::Child => 0.8 * BASE_SPEED,      // Kecepatan rendah
            AgentType::Teen => 1.0 * BASE_SPEED,      // Kecepatan lebih tinggi
            AgentType::Adult => 1.0 * BASE_SPEED,     // Kecepatan sedang 0.75 -> 1.16 m/s 
            AgentType::Elder => 0.7 * BASE_SPEED,     // Kecepatan rendah 0.4 -> 2.5 m/s == 6.25
        } as u32;

        Agent {
            id, // ID akan diatur nanti
            x,
            y,
            speed,
            remaining_steps: speed,
            is_on_road,
            agent_type,
            is_alive: true,
            // --- Initialize new fields ---
            // Use Rng trait for gen_range - Apply deprecated fixes
            knowledge_level: thread_rng().gen_range(10..=90), // Use imported thread_rng
            household_size: thread_rng().gen_range(1..=5), // Use imported thread_rng
            has_decided_to_evacuate: false, // Start undecided
            evacuation_trigger_time: None, // Not triggered initially
            is_in_shelter: false, // Start outside shelter
            milling_steps_remaining: 0, // Start with no milling delay active
        }
    }
}


// Removed: use rand::prelude::IndexedRandom;
// Removed: use rand::{rng, thread_rng}; // We'll use thread_rng() directly where needed

impl AgentType {
    pub fn random() -> Self {
        let weights = [6.21, 13.41, 59.10, 19.89]; // Distribusi bobot

        // Removed unused variants array
        // let variants = [
        //     AgentType::Child,
        //     AgentType::Teen,
        //     AgentType::Adult,
        //     AgentType::Elder,
        // ];
        let mut rng = thread_rng(); // Get thread-local RNG
        let dist = WeightedIndex::new(&weights).unwrap();


        // *variants.choose(&mut rng).unwrap() // variants was removed
        match dist.sample(&mut rng) {
            0 => AgentType::Child,
            1 => AgentType::Teen,
            2 => AgentType::Adult,
            3 => AgentType::Elder,
            _ => AgentType::Adult,
        }
    }
}


use serde::{Serialize, Deserialize};

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
