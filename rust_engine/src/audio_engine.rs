//! Audio engine for real-time playback with DSP

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use crossbeam_channel::{bounded, Receiver, Sender};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use parking_lot::RwLock;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

use crate::dsp::{PitchShifter, TimeStretcher};
use crate::types::{BlobParams, PlaybackState, ProcessingBlob};

/// Commands sent from Python to the audio thread
#[derive(Clone, Debug)]
enum AudioCommand {
    Play,
    Stop,
    Seek(i64),
    SetLoop(Option<i64>, Option<i64>),
    UpdateBlob(BlobParams),
    UpdateAllBlobs(Vec<BlobParams>),
}

/// Audio engine managing real-time playback with pitch/time manipulation
#[pyclass(unsendable)]
pub struct AudioEngine {
    // Audio data (shared with audio thread)
    audio_data: Arc<RwLock<Vec<f32>>>,
    sample_rate: Arc<AtomicI64>,

    // Playback state
    is_playing: Arc<AtomicBool>,
    position: Arc<AtomicI64>,

    // Loop region
    loop_start: Arc<RwLock<Option<i64>>>,
    loop_end: Arc<RwLock<Option<i64>>>,

    // Note blobs for processing
    blobs: Arc<RwLock<Vec<ProcessingBlob>>>,

    // Communication channel
    command_tx: Option<Sender<AudioCommand>>,

    // Audio stream (kept alive)
    _stream: Option<Stream>,
}

#[pymethods]
impl AudioEngine {
    /// Create a new audio engine
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(AudioEngine {
            audio_data: Arc::new(RwLock::new(Vec::new())),
            sample_rate: Arc::new(AtomicI64::new(44100)),
            is_playing: Arc::new(AtomicBool::new(false)),
            position: Arc::new(AtomicI64::new(0)),
            loop_start: Arc::new(RwLock::new(None)),
            loop_end: Arc::new(RwLock::new(None)),
            blobs: Arc::new(RwLock::new(Vec::new())),
            command_tx: None,
            _stream: None,
        })
    }

    /// Load audio data from a numpy array
    pub fn load_audio(&mut self, audio: PyReadonlyArray1<f32>, sample_rate: i64) -> PyResult<()> {
        // Validate inputs
        if sample_rate <= 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sample_rate must be positive",
            ));
        }

        let data: Vec<f32> = audio.as_slice()?.to_vec();
        if data.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "audio data cannot be empty",
            ));
        }

        *self.audio_data.write() = data;
        self.sample_rate.store(sample_rate, Ordering::SeqCst);
        self.position.store(0, Ordering::SeqCst);
        Ok(())
    }

    /// Initialize the audio output stream
    pub fn initialize(&mut self) -> PyResult<()> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No output device"))?;

        let config = device.default_output_config().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to get config: {}",
                e
            ))
        })?;

        let sample_format = config.sample_format();
        let config: StreamConfig = config.into();

        // Create command channel
        let (tx, rx) = bounded::<AudioCommand>(256);
        self.command_tx = Some(tx);

        // Clone shared state for audio thread
        let audio_data = Arc::clone(&self.audio_data);
        let sample_rate = Arc::clone(&self.sample_rate);
        let is_playing = Arc::clone(&self.is_playing);
        let position = Arc::clone(&self.position);
        let loop_start = Arc::clone(&self.loop_start);
        let loop_end = Arc::clone(&self.loop_end);
        let blobs = Arc::clone(&self.blobs);

        let channels = config.channels as usize;
        let device_sample_rate = config.sample_rate.0;

        // Build stream based on sample format
        let stream = match sample_format {
            cpal::SampleFormat::F32 => self.build_stream::<f32>(
                &device,
                &config,
                rx,
                audio_data,
                sample_rate,
                is_playing,
                position,
                loop_start,
                loop_end,
                blobs,
                channels,
                device_sample_rate,
            )?,
            cpal::SampleFormat::I16 => self.build_stream::<i16>(
                &device,
                &config,
                rx,
                audio_data,
                sample_rate,
                is_playing,
                position,
                loop_start,
                loop_end,
                blobs,
                channels,
                device_sample_rate,
            )?,
            cpal::SampleFormat::U16 => self.build_stream::<u16>(
                &device,
                &config,
                rx,
                audio_data,
                sample_rate,
                is_playing,
                position,
                loop_start,
                loop_end,
                blobs,
                channels,
                device_sample_rate,
            )?,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Unsupported sample format",
                ))
            }
        };

        stream.play().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to start stream: {}",
                e
            ))
        })?;

        self._stream = Some(stream);
        Ok(())
    }

    /// Start playback
    pub fn play(&self) -> PyResult<()> {
        self.is_playing.store(true, Ordering::SeqCst);
        if let Some(tx) = &self.command_tx {
            let _ = tx.try_send(AudioCommand::Play);
        }
        Ok(())
    }

    /// Stop playback
    pub fn stop(&self) -> PyResult<()> {
        self.is_playing.store(false, Ordering::SeqCst);
        if let Some(tx) = &self.command_tx {
            let _ = tx.try_send(AudioCommand::Stop);
        }
        Ok(())
    }

    /// Seek to a position in samples
    pub fn seek(&self, position_samples: i64) -> PyResult<()> {
        self.position.store(position_samples, Ordering::SeqCst);
        if let Some(tx) = &self.command_tx {
            let _ = tx.try_send(AudioCommand::Seek(position_samples));
        }
        Ok(())
    }

    /// Set loop region (None to disable)
    pub fn set_loop(&self, start: Option<i64>, end: Option<i64>) -> PyResult<()> {
        *self.loop_start.write() = start;
        *self.loop_end.write() = end;
        if let Some(tx) = &self.command_tx {
            let _ = tx.try_send(AudioCommand::SetLoop(start, end));
        }
        Ok(())
    }

    /// Update a single blob's parameters
    pub fn update_blob(&self, params: BlobParams) -> PyResult<()> {
        {
            let mut blobs = self.blobs.write();
            if let Some(blob) = blobs.iter_mut().find(|b| b.params.id == params.id) {
                blob.params = params.clone();
            }
        }
        if let Some(tx) = &self.command_tx {
            let _ = tx.try_send(AudioCommand::UpdateBlob(params));
        }
        Ok(())
    }

    /// Set all blob parameters at once
    pub fn set_blobs(&self, blobs: Vec<BlobParams>) -> PyResult<()> {
        {
            let mut current_blobs = self.blobs.write();
            *current_blobs = blobs.iter().map(|p| ProcessingBlob::new(p.clone())).collect();
        }
        if let Some(tx) = &self.command_tx {
            let _ = tx.try_send(AudioCommand::UpdateAllBlobs(blobs));
        }
        Ok(())
    }

    /// Get current playback state
    pub fn get_state(&self) -> PlaybackState {
        let sr = self.sample_rate.load(Ordering::SeqCst);
        PlaybackState {
            is_playing: self.is_playing.load(Ordering::SeqCst),
            position_samples: self.position.load(Ordering::SeqCst),
            // Safely convert i64 to u32, clamping to valid range
            sample_rate: sr.clamp(0, u32::MAX as i64) as u32,
            current_blob_id: None, // TODO: track current blob
        }
    }

    /// Get current position in samples
    pub fn get_position(&self) -> i64 {
        self.position.load(Ordering::SeqCst)
    }

    /// Check if currently playing
    pub fn is_playing(&self) -> bool {
        self.is_playing.load(Ordering::SeqCst)
    }

    /// Process the entire audio file offline with current blob settings
    /// and return the result.
    pub fn process_offline(
        &self,
        py: Python,
        blobs: Vec<BlobParams>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let audio_data = self.audio_data.read();
        let mut output = audio_data.clone();

        let mut pitch_shifter = PitchShifter::new();
        let mut time_stretcher = TimeStretcher::new();

        // TODO: This is a simplified offline render. A more robust solution
        // would use a proper graph-based processor to handle overlaps.

        for blob in blobs {
            let start = blob.start_sample as usize;
            let end = blob.end_sample as usize;
            if start >= end || end > audio_data.len() {
                continue;
            }

            let original_len = end - start;
            let mut chunk = audio_data[start..end].to_vec();

            // 1. Time Stretch
            let stretch_ratio = blob.stretch_ratio();
            if (stretch_ratio - 1.0).abs() > 0.01 {
                chunk = time_stretcher.process(&chunk, stretch_ratio);
            }

            // 2. Pitch Shift
            let pitch_ratio = blob.pitch_ratio();
            if (pitch_ratio - 1.0).abs() > 0.01 {
                let mut shifted_chunk = vec![0.0; chunk.len()];
                pitch_shifter.process(&chunk, &mut shifted_chunk, pitch_ratio);
                chunk = shifted_chunk;
            }

            // 3. Overlap-add into output buffer
            // This simple version just replaces the segment, which can cause artifacts.
            let new_len = chunk.len();
            let new_end = start + new_len;
            if new_end > output.len() {
                output.resize(new_end, 0.0);
            }
            // Basic crossfade to reduce clicks
            let fade_len = (self.sample_rate.load(Ordering::SeqCst) as f32 * 0.01) as usize;
            self.crossfade_and_replace(&mut output, &chunk, start, fade_len);
        }

        Ok(output.into_pyarray(py).to_owned())
    }

    /// Get list of available output devices
    #[staticmethod]
    pub fn get_output_devices() -> PyResult<Vec<String>> {
        let host = cpal::default_host();
        let devices: Vec<String> = host
            .output_devices()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to get devices: {}",
                    e
                ))
            })?
            .filter_map(|d| d.name().ok())
            .collect();
        Ok(devices)
    }
}

impl AudioEngine {
    fn build_stream<T: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32>>(
        &self,
        device: &Device,
        config: &StreamConfig,
        rx: Receiver<AudioCommand>,
        audio_data: Arc<RwLock<Vec<f32>>>,
        _source_sample_rate: Arc<AtomicI64>,
        is_playing: Arc<AtomicBool>,
        position: Arc<AtomicI64>,
        loop_start: Arc<RwLock<Option<i64>>>,
        loop_end: Arc<RwLock<Option<i64>>>,
        blobs: Arc<RwLock<Vec<ProcessingBlob>>>,
        channels: usize,
        _device_sample_rate: u32,
    ) -> PyResult<Stream> {
        // DSP processors for the audio thread
        let mut pitch_shifter = PitchShifter::new();
        let mut _time_stretcher = TimeStretcher::new();

        // Processing buffers
        let mut process_buffer = vec![0.0f32; 4096];
        let mut output_buffer = vec![0.0f32; 4096];

        let stream = device
            .build_output_stream(
                config,
                move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                    // Process any pending commands
                    while let Ok(cmd) = rx.try_recv() {
                        match cmd {
                            AudioCommand::Seek(pos) => {
                                position.store(pos, Ordering::SeqCst);
                                pitch_shifter.reset();
                            }
                            AudioCommand::Stop => {
                                pitch_shifter.reset();
                            }
                            _ => {}
                        }
                    }

                    let playing = is_playing.load(Ordering::SeqCst);

                    if !playing {
                        // Output silence when not playing
                        for sample in data.iter_mut() {
                            *sample = T::from_sample(0.0f32);
                        }
                        return;
                    }

                    let audio = audio_data.read();
                    let num_frames = data.len() / channels;

                    // Ensure buffers are large enough (with reasonable max to prevent OOM)
                    const MAX_BUFFER_SIZE: usize = 1024 * 1024; // 1M samples max
                    let safe_num_frames = num_frames.min(MAX_BUFFER_SIZE);
                    if process_buffer.len() < safe_num_frames {
                        process_buffer.resize(safe_num_frames, 0.0);
                        output_buffer.resize(safe_num_frames, 0.0);
                    }

                    // Safely convert position to usize, clamping negative values
                    let pos_i64 = position.load(Ordering::SeqCst);
                    let mut pos = if pos_i64 < 0 { 0usize } else { pos_i64 as usize };
                    let loop_s = *loop_start.read();
                    let loop_e = *loop_end.read();

                    // Get current blob parameters
                    let current_blobs = blobs.read();

                    // Read source audio
                    for i in 0..num_frames {
                        if pos < audio.len() {
                            process_buffer[i] = audio[pos];
                            pos += 1;
                        } else {
                            process_buffer[i] = 0.0;
                        }

                        // Handle looping
                        if let (Some(ls), Some(le)) = (loop_s, loop_e) {
                            if pos as i64 >= le {
                                pos = ls as usize;
                            }
                        }
                    }

                    // Find active blob and apply processing
                    let current_pos = position.load(Ordering::SeqCst);
                    let active_blob = current_blobs.iter().find(|b| {
                        current_pos >= b.params.start_sample && current_pos < b.params.end_sample
                    });

                    if let Some(blob) = active_blob {
                        // Apply pitch shift
                        let pitch_ratio = blob.params.pitch_ratio();
                        if (pitch_ratio - 1.0).abs() > 0.01 {
                            pitch_shifter.process(
                                &process_buffer[..num_frames],
                                &mut output_buffer[..num_frames],
                                pitch_ratio,
                            );
                        } else {
                            output_buffer[..num_frames]
                                .copy_from_slice(&process_buffer[..num_frames]);
                        }
                    } else {
                        // No blob, pass through
                        output_buffer[..num_frames].copy_from_slice(&process_buffer[..num_frames]);
                    }

                    // Write to output
                    for (i, frame) in data.chunks_mut(channels).enumerate() {
                        let sample = if i < num_frames {
                            output_buffer[i]
                        } else {
                            0.0
                        };

                        // Clamp and write to all channels
                        let clamped = sample.clamp(-1.0, 1.0);
                        for channel in frame.iter_mut() {
                            *channel = T::from_sample(clamped);
                        }
                    }

                    // Update position
                    position.store(pos as i64, Ordering::SeqCst);
                },
                |err| {
                    eprintln!("Audio stream error: {}", err);
                },
                None,
            )
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to build stream: {}",
                    e
                ))
            })?;

        Ok(stream)
    }

    /// Overlap-add a chunk into the main buffer with crossfading
    fn crossfade_and_replace(&self, buffer: &mut Vec<f32>, chunk: &[f32], position: usize, fade_len: usize) {
        if position + chunk.len() > buffer.len() {
            buffer.resize(position + chunk.len(), 0.0);
        }

        let safe_fade_len = fade_len.min(chunk.len() / 2).min(position);

        // Fade in
        for i in 0..safe_fade_len {
            let gain = i as f32 / safe_fade_len as f32;
            buffer[position + i] = buffer[position + i] * (1.0 - gain) + chunk[i] * gain;
        }

        // Copy main part
        let main_part_start = position + safe_fade_len;
        let chunk_main_part_end = chunk.len() - safe_fade_len;
        buffer[main_part_start..position + chunk_main_part_end].copy_from_slice(&chunk[safe_fade_len..chunk_main_part_end]);

        // Fade out
        for i in 0..safe_fade_len {
            let gain = i as f32 / safe_fade_len as f32;
            let buf_idx = position + chunk.len() - safe_fade_len + i;
            let chunk_idx = chunk.len() - safe_fade_len + i;
            buffer[buf_idx] = buffer[buf_idx] * gain + chunk[chunk_idx] * (1.0 - gain);
        }
    }
}

impl Default for AudioEngine {
    fn default() -> Self {
        AudioEngine {
            audio_data: Arc::new(RwLock::new(Vec::new())),
            sample_rate: Arc::new(AtomicI64::new(44100)),
            is_playing: Arc::new(AtomicBool::new(false)),
            position: Arc::new(AtomicI64::new(0)),
            loop_start: Arc::new(RwLock::new(None)),
            loop_end: Arc::new(RwLock::new(None)),
            blobs: Arc::new(RwLock::new(Vec::new())),
            command_tx: None,
            _stream: None,
        }
    }
}
