pub mod custom_types;
pub mod llm;
pub mod memory_tools;
pub mod memory_management;
pub mod summary;
pub mod utils;

use std::collections::HashMap;
use std::io::Write;

use tokio::time::{ sleep, Duration };
use tokio::signal;

use lm_studio_api_extended::chat::*;
use lm_studio_api_extended::embedding::*;

use crate::memory_management::{add_memory, retrieve_memory};
use crate::summary::generate_summary;
use crate::utils::{ load_sysprompt, MAX_TOKENS};
use crate::llm::generate_response;


#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    run().await?;
    Ok(())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let sysprompt = load_sysprompt()?;
    let mut context = Context::new(
        sysprompt, 
        MAX_TOKENS as usize
    );

    let mut memories: HashMap<String, Vec<f32>> = HashMap::new();
    let mut memories_text: HashMap<String, String> = HashMap::new();
    let mut count: usize = 0;

    let mut embedder = Embedding::new(None); // Uses default localhost:1234
    
    println!("Async loop started. Press Ctrl+C to stop.");

    tokio::select! {
        _ = run_loop(
            &mut context, &mut memories, &mut memories_text,
            &mut count, &mut embedder) => {},
        _ = signal::ctrl_c() => {
            println!("\nCtrl+C received!");
        }
    }

    println!("Loop stopped.");
    Ok(())
}

async fn run_loop(
    context: &mut Context, memories: &mut HashMap<String, Vec<f32>>,
    memories_text: &mut HashMap<String, String>, count: &mut usize, 
    embedder: &mut Embedding
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        sleep(Duration::from_secs(1)).await;
        print!("Enter your message here: ");
        let mut input: String = String::new();
        std::io::stdout().flush().unwrap();

        std::io::stdin()
            .read_line(&mut input)?;
        println!();
        std::io::stdout().flush().unwrap();

        if !memories_text.len() <= 50 {
            context.add(Message {
                role: Role::User,
                content: input.clone()
            });
        }

        if memories_text.len() > 50 {
            let infos = retrieve_memory(memories.clone(), memories_text.clone(), &input.clone(), 3, embedder).await?;

            let modification: String = if infos.len() > 1 {
                infos.iter()
                    .map(|i| format!("{}\n\t", i))
                    .collect::<String>()
            } else {
                format!("{}\n\t", infos[0]).to_string()
            };

            context.edit(modification);

            context.add(Message {
                role: Role::User,
                content: input.clone()
            })
        }

        
        let completion = generate_response(context.clone()).await?;
        add_memory(&input, memories, memories_text, count, embedder).await?;
        
        println!("Response: {:#?}", completion);
        println!("Memories as text length: {}", memories_text.len());
        println!("Memories as embeds length: {}", memories.len());
        println!("Summary: {}", generate_summary(&input).await?)
    }
}