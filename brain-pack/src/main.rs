//! brain-pack CLI entry point.
//!
//! Delegates to brain_pack::run_cli.

use brain_pack::run_cli;

fn main() {
    if let Err(e) = run_cli() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
