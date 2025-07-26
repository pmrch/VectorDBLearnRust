use log::{self, info};

use openai_api_rust::completions::Completion;
use openai_api_rust::*;
use openai_api_rust::chat::*;

use crate::custom_types::{ MyChatBody, MyChatbot };
use crate::utils::{ CHAT_MODEL };

static MY_ZERO: Option<u32> = Some(0);
static MY_STR: &str = "";


impl MyChatbot {
    pub fn new() -> Self {
        MyChatbot
    }

    pub fn generate_response(self, messages: &mut Vec<Message>, openai: &OpenAI) -> (String, Option<u32>, Option<u32>) {
        let oai = openai.clone();
        let max_tokens: i32 = 512;
        let temperature: f32 = 0.3;

        info!("Creating ChatBdy...");
        let body = MyChatBody::new(CHAT_MODEL, messages.to_vec())
            .with_max_tokens(max_tokens)
            .with_temperature(temperature);
        info!("Created ChatBody");
            
        info!("Trying to get result");
        let result = match oai.chat_completion_create(&body) {
            Ok(res) => {
                info!("Completion successfully generate");
                res
            }
            Err(e) => {
                log::error!("Failed to get result for API error: {e}");
                Completion {
                    id: None,
                    object: None,
                    created: None,
                    model: None,
                    choices: vec![],
                    usage: Usage { 
                        prompt_tokens: None, 
                        completion_tokens: None, 
                        total_tokens: None 
                    }
                }
            }
        };

        if result.choices.len() == 0 { return (MY_STR.to_string(), MY_ZERO, MY_ZERO); }

        let choices = &result.choices;
        let message = choices[0].message.as_ref().unwrap();
        let true_msg = message.content.clone();

        let usage = &result.usage;
        let user_usage: Option<u32> = usage.prompt_tokens;
        let agent_usage: Option<u32> = usage.completion_tokens;

        //println!("\n\nMessage: {}", true_msg);
        //println!("User Usage: {:#?}\nAgent Usage: {:#?}", user_usage, agent_usage);
        return (true_msg, user_usage, agent_usage);

    }
}