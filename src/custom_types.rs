use log::{self, Level, Metadata, Record};

use openai_api_rust::embeddings::{ EmbeddingsBody, EmbeddingData };
use openai_api_rust::Usage;
use openai_api_rust::chat::ChatBody;
use openai_api_rust::Message;




#[derive(Clone)]
pub struct MyEmbeddings {
	pub object: Option<String>,
	pub data: Option<Vec<EmbeddingData>>,
	pub model: String,
	pub usage: Usage,
}

// Here we predefine the MyChatbot struct, and implement
// methods elsewhere
#[derive(Clone)]
pub struct MyChatbot;

// This struct is created without parameters, because we only
// want to implement methods for it
pub struct  MyLogger;

// Here we implement the log::Log trait to MyLogger to support
// the same operations like warn! info! error!
impl log::Log for MyLogger {
    // Here we create the first required method for the struct
    // This tells us whether or not to log
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    // Here we create the second required method for the struct
    // This basically does the logging itself
    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            println!("{} - {}", record.level(), record.args());
        }
    }

    // Here we create the third required method for the struct
    // This can be left empty
    fn flush(&self) {}
}


// This is a public struct, accessible from any other
// source code file.
// It creates MyChatBody, taking ownership of ChatBody to 
// allow implementations
pub struct MyChatBody(ChatBody);

// Implements the `::new()` constructor for MyChatBody,
// a simplified wrapper over ChatBody for easier creation.
impl MyChatBody {
    // This is the ::new() attribute for the struct MyChatBody,
    // basically like creating a method inside a class in high-level
    // programming languages
    pub fn new(model: &str, messages: Vec<Message>) -> Self {
        // The purpose of this function is to simplify ChatBody construction
        // anywhere in the codebase, because a lot of parameters must be defined
        // when defining a ChatBody, most of which is unused, hence the None type.
        // So we only have to provide a few common parameters
        MyChatBody(ChatBody { 
            model: model.to_string(), 
            messages, 
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            max_tokens: None,
            presence_penalty: None, 
            frequency_penalty: None, 
            logit_bias: None, 
            user: None 
        })
    }

    // This function allows chaining methods to the constructor,
    // in this case changing the max_tokens value of the struct
    // MyChatBody.
    pub fn with_max_tokens(mut self, max_tokens: i32) -> Self {
        self.0.max_tokens = Some(max_tokens);
        self
    }

    // This function does the same as the method above, but in this
    // case we are modifying the temperature value.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.0.temperature = Some(temperature);
        self
    }

    // Since MyChatBot is a wrapper for ChatBody, we can unwrap it if we want
    // to use ChatBody instead of MyChatBody
    // Idiomatically we call this into_inner(), but we also call it unwrap()
    pub fn into_inner(self) -> ChatBody {
        self.0
    }
}

/*  Also, Self with capital S refers to the struct we apply the
    implementation to, in this case MyChatBody.

    self with lowercase s refers to the instance of the sctruct defined
    as a parameter of the method */

impl std::ops::Deref for MyChatBody {
    type Target = ChatBody;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}


// Here we create a struct that acts as a wrapper for
// EmbeddingsBody, which we own, not borrow
pub struct MyEmbeddingBody(EmbeddingsBody);

impl MyEmbeddingBody {
    pub fn new(model: &str, input: Vec<String>) -> Self {
        // The purpose of this function is to simplify EmbeddingsBody construction
        // anywhere in the codebase, because a lot of parameters must be defined
        // when defining a ChatBody, most of which is unused, hence the None type.
        // So we only have to provide a few common parameters
        MyEmbeddingBody(EmbeddingsBody { 
            model: model.to_string(),
            input,
            user: None
        })
    }

    // Since MyChatBot is a wrapper for ChatBody, we can unwrap it if we want
    // to use ChatBody instead of MyChatBody
    // Idiomatically we call this into_inner(), but we also call it unwrap()
    pub fn into_inner(self) -> EmbeddingsBody {
        self.0
    }
}

impl std::ops::Deref for MyEmbeddingBody {
    type Target = EmbeddingsBody;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}