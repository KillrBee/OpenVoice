//! Digital Signal Processing utilities
//!
//! Provides pitch shifting and time stretching functionality.

use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// Size of FFT for spectral processing
const FFT_SIZE: usize = 2048;
const HOP_SIZE: usize = FFT_SIZE / 4;

/// Phase vocoder for pitch shifting
pub struct PitchShifter {
    fft_size: usize,
    hop_size: usize,
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn realfft::ComplexToReal<f32>>,
    window: Vec<f32>,
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    last_phase: Vec<f32>,
    sum_phase: Vec<f32>,
    spectrum: Vec<num_complex::Complex<f32>>,
    scratch: Vec<num_complex::Complex<f32>>,
}

impl PitchShifter {
    pub fn new() -> Self {
        let fft_size = FFT_SIZE;
        let hop_size = HOP_SIZE;

        let mut planner = RealFftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);

        // Hann window
        let window: Vec<f32> = (0..fft_size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos())
            })
            .collect();

        let spectrum_len = fft_size / 2 + 1;

        PitchShifter {
            fft_size,
            hop_size,
            fft,
            ifft,
            window,
            input_buffer: vec![0.0; fft_size],
            output_buffer: vec![0.0; fft_size * 2],
            last_phase: vec![0.0; spectrum_len],
            sum_phase: vec![0.0; spectrum_len],
            spectrum: vec![num_complex::Complex::new(0.0, 0.0); spectrum_len],
            scratch: vec![num_complex::Complex::new(0.0, 0.0); spectrum_len],
        }
    }

    /// Reset internal state (call when seeking)
    pub fn reset(&mut self) {
        self.input_buffer.fill(0.0);
        self.output_buffer.fill(0.0);
        self.last_phase.fill(0.0);
        self.sum_phase.fill(0.0);
    }

    /// Process a block of audio with pitch shift
    ///
    /// # Arguments
    /// * `input` - Input samples
    /// * `output` - Output buffer (must be same size as input)
    /// * `pitch_ratio` - Pitch shift ratio (1.0 = no shift, 2.0 = octave up)
    pub fn process(&mut self, input: &[f32], output: &mut [f32], pitch_ratio: f32) {
        if pitch_ratio == 1.0 {
            // No shift needed, just copy
            output[..input.len()].copy_from_slice(input);
            return;
        }

        output.fill(0.0);

        let num_frames = (input.len() / self.hop_size).saturating_sub(3);

        for frame in 0..num_frames {
            let in_offset = frame * self.hop_size;

            // Copy input with window
            for i in 0..self.fft_size {
                if in_offset + i < input.len() {
                    self.input_buffer[i] = input[in_offset + i] * self.window[i];
                } else {
                    self.input_buffer[i] = 0.0;
                }
            }

            // Forward FFT
            if let Err(_) = self.fft.process(&mut self.input_buffer, &mut self.spectrum) {
                // FFT failed, skip this frame
                continue;
            }

            // Phase vocoder pitch shifting
            self.pitch_shift_spectrum(pitch_ratio);

            // Inverse FFT
            if let Err(_) = self.ifft.process(&mut self.spectrum, &mut self.input_buffer) {
                // IFFT failed, skip this frame
                continue;
            }

            // Normalize and overlap-add
            let norm = 1.0 / (self.fft_size as f32);
            let out_offset = frame * self.hop_size;

            for i in 0..self.fft_size {
                let out_idx = out_offset + i;
                if out_idx < output.len() {
                    output[out_idx] += self.input_buffer[i] * self.window[i] * norm;
                }
            }
        }

        // Normalize overlap-add
        let ola_factor = self.fft_size as f32 / self.hop_size as f32 / 2.0;
        for sample in output.iter_mut() {
            *sample /= ola_factor;
        }
    }

    fn pitch_shift_spectrum(&mut self, pitch_ratio: f32) {
        let freq_per_bin = 1.0 / self.fft_size as f32;
        let expect = 2.0 * std::f32::consts::PI * self.hop_size as f32 / self.fft_size as f32;

        // Analyze phases
        for k in 0..self.spectrum.len() {
            let mag = self.spectrum[k].norm();
            let phase = self.spectrum[k].arg();

            // Get phase difference
            let mut phase_diff = phase - self.last_phase[k];
            self.last_phase[k] = phase;

            // Subtract expected phase
            phase_diff -= k as f32 * expect;

            // Wrap to -pi..pi
            phase_diff = phase_diff - (phase_diff / (2.0 * std::f32::consts::PI)).round()
                * 2.0
                * std::f32::consts::PI;

            // Get true frequency
            let true_freq = k as f32 * freq_per_bin + phase_diff / expect * freq_per_bin;

            // Store magnitude and frequency for resynthesis
            self.scratch[k] = num_complex::Complex::new(mag, true_freq);
        }

        // Clear spectrum for synthesis
        self.spectrum.fill(num_complex::Complex::new(0.0, 0.0));

        // Resynthesize with shifted frequencies
        for k in 0..self.scratch.len() {
            let new_bin = (k as f32 * pitch_ratio).round() as usize;
            if new_bin < self.spectrum.len() {
                let mag = self.scratch[k].re;
                let true_freq = self.scratch[k].im;

                // Shift frequency
                let shifted_freq = true_freq * pitch_ratio;

                // Accumulate phase
                self.sum_phase[new_bin] +=
                    expect * (shifted_freq / freq_per_bin - new_bin as f32);
                self.sum_phase[new_bin] += new_bin as f32 * expect;

                // Convert back to complex
                self.spectrum[new_bin] = num_complex::Complex::from_polar(mag, self.sum_phase[new_bin]);
            }
        }
    }
}

impl Default for PitchShifter {
    fn default() -> Self {
        Self::new()
    }
}

/// Time stretcher using overlap-add
pub struct TimeStretcher {
    fft_size: usize,
    window: Vec<f32>,
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
}

impl TimeStretcher {
    pub fn new() -> Self {
        let fft_size = FFT_SIZE;

        // Hann window
        let window: Vec<f32> = (0..fft_size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos())
            })
            .collect();

        TimeStretcher {
            fft_size,
            window,
            input_buffer: vec![0.0; fft_size],
            output_buffer: vec![0.0; fft_size * 4],
        }
    }

    /// Process audio with time stretch using WSOLA-like algorithm
    ///
    /// # Arguments
    /// * `input` - Input samples
    /// * `stretch_ratio` - Time stretch ratio (1.0 = original, 2.0 = twice as long)
    ///
    /// # Returns
    /// Stretched audio samples
    pub fn process(&mut self, input: &[f32], stretch_ratio: f32) -> Vec<f32> {
        if (stretch_ratio - 1.0).abs() < 0.01 {
            // No stretch needed
            return input.to_vec();
        }

        let output_len = (input.len() as f32 * stretch_ratio) as usize;
        let mut output = vec![0.0; output_len];

        let analysis_hop = self.fft_size / 4;
        let synthesis_hop = (analysis_hop as f32 * stretch_ratio) as usize;

        let num_frames = input.len() / analysis_hop;
        let mut write_pos = 0;

        for frame in 0..num_frames {
            let read_pos = frame * analysis_hop;

            // Copy input frame with window
            for i in 0..self.fft_size {
                let idx = read_pos + i;
                if idx < input.len() {
                    self.input_buffer[i] = input[idx] * self.window[i];
                } else {
                    self.input_buffer[i] = 0.0;
                }
            }

            // Overlap-add to output
            for i in 0..self.fft_size {
                let out_idx = write_pos + i;
                if out_idx < output.len() {
                    output[out_idx] += self.input_buffer[i] * self.window[i];
                }
            }

            write_pos += synthesis_hop;
        }

        // Normalize
        let norm_factor = self.fft_size as f32 / synthesis_hop as f32 / 2.0;
        for sample in output.iter_mut() {
            *sample /= norm_factor;
        }

        output
    }

    pub fn reset(&mut self) {
        self.input_buffer.fill(0.0);
        self.output_buffer.fill(0.0);
    }
}

impl Default for TimeStretcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pitch_shifter_passthrough() {
        let mut shifter = PitchShifter::new();
        let input: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut output = vec![0.0; input.len()];

        shifter.process(&input, &mut output, 1.0);

        // Should be approximately equal for ratio 1.0
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 0.01, "Passthrough failed");
        }
    }

    #[test]
    fn test_time_stretcher_identity() {
        let mut stretcher = TimeStretcher::new();
        let input: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect();

        let output = stretcher.process(&input, 1.0);

        assert_eq!(input.len(), output.len());
    }
}
