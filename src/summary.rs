use lm_studio_api_extended::chat::*;

use crate::utils::{ MAX_TOKENS, TEMPERATURE };

pub async fn generate_summary(input: &str) -> std::result::Result<String, Box<dyn std::error::Error>> {
    let sysprompt = { 
        "You are a helpful assistant that summarizes what users say into brief factual 
        statements, written in the third person. Avoid emotional commentary or repetition. 
        Each fact should be a complete sentence and clearly describe what the user said. 
        If the input is long or rambling, extract only the important information.
        
        
        **Always refer to the user as \"the user\", never assume gender.**" 
    };

    let mut context: Context = Context::new(
        sysprompt, 
        256
    );
    context.add(
        Message {
            role: Role::User,
            content: input.to_string()
        }
    );

    let mut chat: Chat = Chat::new(
        Model::Llama3_1_8b, 
        context.clone(), 
        "1234"
    );

    let my_request = Request {
        model: Model::Llama3_1_8b,
        messages: context.clone().get(),
        context: true,
        temperature: TEMPERATURE,
        max_tokens: MAX_TOKENS,
        stream: false
    };

    let response = chat.send(my_request).await?;
    if let Some(resp) = response {
        std::result::Result::Ok(resp.text().to_string())
    } else {
        std::result::Result::Err("Chat response creation failed".into())
    }
}