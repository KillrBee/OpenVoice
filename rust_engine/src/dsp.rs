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
    pub fn process(&mut self, input: &[f32], output: &mut [f32], pitch_ratio: f32) {
        if (pitch_ratio - 1.0).abs() < 1e-4 {
            output[..input.len()].copy_from_slice(input);
            return;
        }

        output.fill(0.0);
        let num_frames = if self.hop_size > 0 { (input.len() / self.hop_size).saturating_sub(3) } else { 0 };

        for frame in 0..num_frames {
            let in_offset = frame * self.hop_size;
            for i in 0..self.fft_size {
                self.input_buffer[i] = input.get(in_offset + i).unwrap_or(&0.0) * self.window[i];
            }

            if self.fft.process(&mut self.input_buffer, &mut self.spectrum).is_err() { continue; }
            self.pitch_shift_spectrum(pitch_ratio);
            if self.ifft.process(&mut self.spectrum, &mut self.input_buffer).is_err() { continue; }

            let norm = 1.0 / (self.fft_size as f32 * 0.5);
            let out_offset = frame * self.hop_size;
            for i in 0..self.fft_size {
                if let Some(out) = output.get_mut(out_offset + i) {
                    *out += self.input_buffer[i] * self.window[i] * norm;
                }
            }
        }
    }

    fn pitch_shift_spectrum(&mut self, pitch_ratio: f32) {
        let freq_per_bin = 1.0 / self.fft_size as f32;
        let expect = 2.0 * std::f32::consts::PI * self.hop_size as f32 / self.fft_size as f32;

        for k in 0..self.spectrum.len() {
            let mag = self.spectrum[k].norm();
            let phase = self.spectrum[k].arg();
            let mut phase_diff = phase - self.last_phase[k];
            self.last_phase[k] = phase;
            phase_diff -= k as f32 * expect;
            phase_diff -= (phase_diff / std::f32::consts::PI).round() * std::f32::consts::PI;
            let true_freq = k as f32 * freq_per_bin + phase_diff / expect * freq_per_bin;
            self.scratch[k] = num_complex::Complex::new(mag, true_freq);
        }

        self.spectrum.fill(num_complex::Complex::new(0.0, 0.0));

        for k in 0..self.scratch.len() {
            let new_bin = (k as f32 * pitch_ratio).round() as usize;
            if new_bin < self.spectrum.len() {
                let mag = self.scratch[k].re;
                let true_freq = self.scratch[k].im;
                let shifted_freq = true_freq * pitch_ratio;
                self.sum_phase[new_bin] += expect * (shifted_freq / freq_per_bin - new_bin as f32);
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

/// Time stretcher using WSOLA algorithm
pub struct TimeStretcher {
    win_size: usize,
    hop_size: usize,
    window: Vec<f32>,
    tolerance: usize,
}

impl TimeStretcher {
    pub fn new() -> Self {
        let win_size = 1024;
        let hop_size = win_size / 4;
        let window: Vec<f32> = (0..win_size)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / win_size as f32).cos()))
            .collect();
        TimeStretcher {
            win_size,
            hop_size,
            window,
            tolerance: hop_size / 2,
        }
    }

    pub fn process(&mut self, input: &[f32], stretch_ratio: f32) -> Vec<f32> {
        if (stretch_ratio - 1.0).abs() < 1e-4 {
            return input.to_vec();
        }

        let output_len = (input.len() as f32 * stretch_ratio) as usize;
        let mut output = vec![0.0; output_len];
        let mut output_pos = 0.0;
        let mut input_pos = 0;

        while input_pos + self.win_size < input.len() && (output_pos as usize) + self.win_size < output_len {
            let best_offset = self.find_best_offset(&input[input_pos..]);
            input_pos += best_offset;

            let input_segment = &input[input_pos..input_pos + self.win_size];
            let output_segment = &mut output[output_pos as usize..output_pos as usize + self.win_size];

            for i in 0..self.win_size {
                output_segment[i] += input_segment[i] * self.window[i];
            }

            output_pos += self.hop_size as f32 * stretch_ratio;
            input_pos += (self.hop_size as f32 * stretch_ratio) as usize;
        }

        output
    }

    fn find_best_offset(&self, input_segment: &[f32]) -> usize {
        let mut max_corr = -1.0;
        let mut best_offset = 0;

        for offset in 0..self.tolerance {
            let mut corr = 0.0;
            let segment1 = &input_segment[offset..offset + self.hop_size];
            let segment2 = &input_segment[self.hop_size..self.hop_size * 2];

            for i in 0..self.hop_size {
                corr += segment1[i] * segment2[i];
            }

            if corr > max_corr {
                max_corr = corr;
                best_offset = offset;
            }
        }
        best_offset
    }

    pub fn reset(&mut self) {}
}

impl Default for TimeStretcher {
    fn default() -> Self {
        Self::new()
    }
}
