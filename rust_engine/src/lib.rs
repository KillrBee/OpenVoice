//! OpenVocal Audio Engine
//!
//! High-performance Rust audio engine for real-time playback with
//! pitch shifting and time stretching capabilities.

use pyo3::prelude::*;

mod audio_engine;
mod dsp;
mod types;

use audio_engine::AudioEngine;
use types::{BlobParams, PlaybackState};

/// OpenVocal Audio Engine Python module
#[pymodule]
fn openvocal_engine(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<AudioEngine>()?;
    m.add_class::<BlobParams>()?;
    m.add_class::<PlaybackState>()?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
