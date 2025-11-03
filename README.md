# zlw — Zero‑Latency Whitening utilities

zlw is a small, focused package that provides zero‑latency whitening utilities for gravitational‑wave data analysis. It includes:

- Whitening filter utilities
  - Minimum‑phase, zero‑latency whitening filters and helpers
  - Supporting Fourier/window helpers for stable, real‑time responses
- MP–MP scheme PSD drift correction terms
  - Utilities to compute first‑ and second‑order timing and phase correction terms
  - Tools to account for slow PSD mismatches between template and data

Where things live:
- zlw.kernels: minimum‑phase whitening filter construction and frequency‑response utilities
- zlw.fourier, zlw.window: helpers used by the whitening filters
- zlw.corrections: MP–MP scheme PSD drift correction terms (class MPMPCorrection)
- zlw/tests: basic tests (e.g., tests/test_kernels.py)
- zlw/src/zlw/bin: small simulation/QA scripts

Quick examples
- Whitening filter
  
  from zlw.kernels import MPWhiteningFilter
  
  # psd: one‑sided PSD array (Hz^-1), fs: sampling rate (Hz), n_fft: FFT length
  wf = MPWhiteningFilter(psd, fs, n_fft)
  Wf = wf.frequency_response()  # one‑sided frequency response (complex for min‑phase)

- MP–MP correction terms
  
  import numpy as np
  from zlw.corrections import MPMPCorrection
  
  # freqs: one‑sided frequency grid; psd1: data PSD; psd2: template PSD; htilde: template FFT
  corr = MPMPCorrection(freqs=freqs, psd1=psd1, psd2=psd2, htilde=htilde, fs=fs)
  # Simple first‑order corrections
  dt1, dphi1 = corr.simplified_correction()
  # Or the full second‑order set (includes cross terms)
  (dt1, dphi1), (dt2, dphi2) = corr.full_correction()

Notes
- “Zero‑latency” refers to the use of minimum‑phase whitening filters so that the whitening operation does not introduce group delay in the time domain.
- The MP–MP correction utilities follow the perturbative scheme that expands about the ratio of PSDs, providing drift terms for coalescence time and phase when the whitening filters differ slightly.
