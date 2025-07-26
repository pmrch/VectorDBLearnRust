use log::*;
use lm_studio_api::*;

use crate::utils::{ CHAT_MODEL, PORT, TEMPERATURE, MAX_TOKENS };


pub async fn generate_response(context: Context) -> Result<Response> {
    let model = Model::Custom(CHAT_MODEL.to_string());
    
    let mut chat: Chat = Chat::new(
        model.clone(), 
        context.clone(), 
        PORT
    );

    let my_request = Request {
        model: model.clone(),
        messages: context.clone().get(),
        context: true,
        temperature: TEMPERATURE,
        max_tokens: MAX_TOKENS,
        stream: false
    };

    let result = chat.send(my_request).await?;
    match result {
        Some(resp) => {
            info!("Successfully generate response with length: {}", resp.choices[0].message.content.len());
            Ok(resp)
        },
        None => {
            warn!("Empty response received.");
            Err(anyhow::anyhow!("Failed to get result").into())
        }
    }

}