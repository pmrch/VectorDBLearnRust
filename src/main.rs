pub mod custom_types;
pub mod llm;
pub mod memory;
pub mod store;
pub mod summary;
pub mod utils;

use std::collections::{HashMap};
use std::io::Write;
use std::sync::atomic::{ AtomicBool, Ordering };
use std::sync::Arc;

use dotenv::dotenv;
use log::{self, info};
use openai_api_rust::embeddings::EmbeddingData;
use openai_api_rust::{Message, Role};

use crate::custom_types::{ MyChatbot };
use crate::utils::{ init_logger, /*text_to_vec*/ load_sysprompt };
use crate::store::add_memory;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        log::error!("\nReceived Ctrl+C, exiting...");
        r.store(false, Ordering::SeqCst);
        std::process::exit(1);
    }).expect("Error setting Ctrl+C handler");

    dotenv().ok();
    match init_logger() {
        Ok(_) => (),
        Err(e) => println!("AN ERROR PREVENTED LOG INITIALIZATION: {e}")
    }

    let cb = MyChatbot::new();
    let system_prompt = load_sysprompt();
    let mut messages: Vec<Message> = vec![
        Message {
            role: Role::System,
            content: match system_prompt {
                Ok(p) if !p.is_empty() => p,
                Ok(_) => {
                    info!("System prompt was successfully loaded, but empty");
                    "".to_string()
                }
                Err(_) => {
                    log::warn!("System Prompt was empty, resuming with an empty one");
                    "".to_string()
                }
            }
        }
    ];
    let mut input: String = String::new();
    let mut memories: HashMap<String, EmbeddingData> = HashMap::new();
    let mut count: usize = 0;

    // Start of chatbot operations
    while running.load(Ordering::SeqCst) {
        print!("Enter your message here: ");
        let _ = std::io::stdout().flush();

        match std::io::stdin().read_line(&mut input) {
            Ok(_) => {
                log::info!("Line read was successful");
            }
            Err(e) => {
                log::error!("Failed to read line due to error: {e}")
            }
        }

        if !input.is_empty() {
            messages.push(
                Message { 
                    role: Role::User, 
                    content: input.to_string()
                }
            );
            add_memory(&input, &mut memories, &mut count);
        }
        
        let resp: (String, Option<u32>, Option<u32>) = cb.clone().generate_response(&mut messages.clone());
        println!("{:#?}", resp.0);
        
        /*for item in text_to_vec(&input) {
            println!("{}", item);
            std::io::stdout().flush().unwrap();
        }*/
    }

    Ok(())
    // End of Chatbot operations
}