use std::{env, fs, io::{ErrorKind, Read}, vec};
use log::*;
use std::path::Path;
//use regex::*;
use openai_api_rust::{Auth, OpenAI};

use crate::custom_types::MyLogger;

//static SPLIT_PAT1: &str = r"^[A-Z].*[.?!]$";
static LOGGER: MyLogger = MyLogger;
pub static EMBED_MODEL: &str = "text-embedding-all-minilm-l6-v2-embedding";
pub static CHAT_MODEL: &str = "meta-llama-3.1-8b-instruct@q4_k_m";

fn read_abbreviations() -> Vec<String> {
    let mut result: Vec<String> = vec![];
    let mut contents: String = String::new();

    let f: Option<fs::File> = match fs::File::open("data/abbreviations.txt") {
        Ok(file) => Some(file),
        Err(e) => {
            error!("Failed to read abbreviations from file: {e}");
            None
        }
    };

    if f.is_none() {
        return vec![];
    }
        
    match f.unwrap().read_to_string(&mut contents) {
        Ok(bytes) => { info!("Read abbreviations with file size: {}", bytes); } ,
        Err(e) => { error!("Failed to run read_to_string due to error: {e}"); }
    }

    for line in contents.split('\n') {
        result.push(line.to_string());
    }

    return result;
}

fn split_to_sentences(input: &str) -> Vec<String> {
    let abbreviations = read_abbreviations();
    let spaced: Vec<(usize, &str)> = input.split_whitespace().enumerate().collect();
    let len = spaced.len();

    let mut result: Vec<String> = vec![];
    let mut buffer: String = String::new();
    
    for (i, s) in &spaced {
        buffer.push_str(s);
        buffer.push(' ');

        let ends_with_punct = s.ends_with('.') || s.ends_with('!') || s.ends_with('?');
        let is_abbreviation = abbreviations.contains(&s.to_string());
        
        if ends_with_punct && !is_abbreviation {
            if *i + 1 < len {
                let next_word = spaced[*i + 1].1;
                let next_w_chars: Vec<char> = next_word.chars().collect();
                
                if let Some(first) = next_w_chars.get(0) {
                    if first.is_uppercase() {
                        result.push(buffer.trim_end().to_string());
                        buffer.clear();
                    }
                }
            }
        }
    } 
    
    if !buffer.trim().is_empty() {
        // Push any leftover text as last sentence
        result.push(buffer.trim_end().to_string());
        buffer.clear();
    }

    return result;
}

pub fn int_to_usize(input: i32) -> Option<usize> {
    if input >= 0 { Some(input as usize) }
    else { None }
}

pub fn load_environment(what: &str) -> String {
    match env::var(what){
        Ok(val) => {
            info!("Successfully read environment variable {what}");
            val
        }
        Err(e) => {
            warn!("Couldn't read environment variable {what}: {e}");
            "".to_string()
        }
    }
}

pub fn load_sysprompt() -> Result<String, std::io::Error> {
    let mut contents: String = String::new();
    let path: &str = "data/system_pormpt.txt";

    let mut f: fs::File = fs::File::open(path)
        .inspect(|_| info!("Successfully read system prompt"))
        .map_err(|err| {
            // error!("Failed to read system prompt due to error {}", err);
            match err.kind() {
                ErrorKind::NotFound => {
                    warn!("System prompt file was missing, trying to create it...");
                    let _ = fs::File::create(path)
                        .inspect(|_| info!("Missing file was successfully created at {}.", path))
                        .map_err(|er| {
                            match er.kind() {
                                ErrorKind::NotFound => {
                                    warn!("The directory of the file was not found, creating it...");
                                    if let Some(parent) = Path::new(path).parent() {
                                        let _ = fs::create_dir_all(parent)
                                            .inspect(|_| {
                                                info!("Successfully recovered from missing directory error");
                                                info!("Trying to create file in recovered directory...");
                                                fs::File::create(path).unwrap();
                                            })
                                            .inspect_err(|err| error!("Critical error prevented directory creation: {}", err));
                                    }
                                },
                                _ => {
                                    error!("Failed to create file due to error: {}", er);
                                }
                            }
                        });
                    },
                other_error => {
                    error!("An unexpected error occured while reading system prompt: {}", other_error);
                }
            }

            err
        })?;
        

    let _ = f.read_to_string(&mut contents)
        .inspect(|val| info!("Successfully read system prompt into string, with size of {} bytes", val))
        .inspect_err(|err| error!("Failed to read system prompt into string due to error: {}", err));

    //if contents.is_empty() { return Err(ErrorKind::No); }
    
    return Ok(contents);
}

pub fn init_logger() -> Result<(), SetLoggerError> {
    log::set_logger(&LOGGER)?;
    log::set_max_level(LevelFilter::Debug);

    Ok(())
}

pub fn get_openai(url: &str, a_key: &str) -> OpenAI {
    let auth = Auth{
        api_key: a_key.to_string(),
        organization: None
    };
    let oai = OpenAI::new(auth, url);

    oai
}

pub fn text_to_vec(input: &str) -> Vec<String>{
    let sentences = split_to_sentences(input);
    sentences
}