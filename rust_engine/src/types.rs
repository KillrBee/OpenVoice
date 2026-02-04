//! Data types shared between Python and Rust

use pyo3::prelude::*;

/// Parameters for a single note blob
#[pyclass]
#[derive(Clone, Debug)]
pub struct BlobParams {
    #[pyo3(get, set)]
    pub id: i64,
    #[pyo3(get, set)]
    pub start_sample: i64,
    #[pyo3(get, set)]
    pub end_sample: i64,
    #[pyo3(get, set)]
    pub shift_semitones: f32,
    #[pyo3(get, set)]
    pub stretch_ratio: f32,
}

#[pymethods]
impl BlobParams {
    #[new]
    pub fn new(
        id: i64,
        start_sample: i64,
        end_sample: i64,
        shift_semitones: f32,
        stretch_ratio: f32,
    ) -> PyResult<Self> {
        // Validate inputs
        if start_sample < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "start_sample must be non-negative",
            ));
        }
        if end_sample < start_sample {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "end_sample must be >= start_sample",
            ));
        }
        if stretch_ratio <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "stretch_ratio must be positive",
            ));
        }

        Ok(BlobParams {
            id,
            start_sample,
            end_sample,
            shift_semitones,
            stretch_ratio,
        })
    }

    /// Get the pitch shift ratio (frequency multiplier)
    pub fn pitch_ratio(&self) -> f32 {
        2.0_f32.powf(self.shift_semitones / 12.0)
    }

    /// Get the number of samples in this blob
    pub fn num_samples(&self) -> i64 {
        self.end_sample - self.start_sample
    }
}

/// Current playback state
#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct PlaybackState {
    #[pyo3(get)]
    pub is_playing: bool,
    #[pyo3(get)]
    pub position_samples: i64,
    #[pyo3(get)]
    pub sample_rate: u32,
    #[pyo3(get)]
    pub current_blob_id: Option<i64>,
}

#[pymethods]
impl PlaybackState {
    #[new]
    pub fn new() -> Self {
        PlaybackState::default()
    }

    /// Get position in seconds
    pub fn position_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.position_samples as f64 / self.sample_rate as f64
    }
}

/// Internal struct for blob processing (not exposed to Python)
#[derive(Clone, Debug)]
pub struct ProcessingBlob {
    pub params: BlobParams,
    pub processed_position: i64, // Position within the blob being processed
}

impl ProcessingBlob {
    pub fn new(params: BlobParams) -> Self {
        ProcessingBlob {
            params,
            processed_position: 0,
        }
    }

    pub fn reset(&mut self) {
        self.processed_position = 0;
    }
}
