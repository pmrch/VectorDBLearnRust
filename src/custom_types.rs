use log::{self, Level, Metadata, Record};

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