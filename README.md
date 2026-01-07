# zlw — Zero‑Latency Whitening utilities

[![Pipeline Status](https://git.ligo.org/james.kennington/zlw/badges/main/pipeline.svg)](https://git.ligo.org/james.kennington/zlw/-/pipelines)
[![Coverage](https://git.ligo.org/james.kennington/zlw/badges/main/coverage.svg)](https://git.ligo.org/james.kennington/zlw/-/graphs/main/charts)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)

**zlw** is a focused package for zero‑latency whitening in gravitational‑wave data analysis. It provides rigorous tools
for:

- **Whitening Kernels:** Computing Minimum‑Phase (MP) filters that whiten data without introducing acausal latency.
- **Perturbative Corrections:** Geometric drift correction terms for the MP–MP scheme. These calculate the precise
  timing ($dt$) and phase ($d\phi$) biases introduced when the template PSD differs slightly from the real noise PSD.

## Installation

```bash
pip install zlw
```

## Structure

- `zlw.kernels`: Construction of MP whitening filters and frequency response utilities.
- `zlw.corrections`: Perturbative MP–MP correction terms (PerturbativeMPCorrection).
- `zlw.fourier`, `zlw.window`: FFT and windowing backend helpers.
- `zlw.bin`: Simulation and QA scripts.

### Part 3: Example 1 (Whitening Filters)

## Quick Examples

### 1. Computing a Whitening Filter

```python
from zlw.kernels import MPWhiteningFilter

# psd: One‑sided PSD array (Hz^-1)
# fs: Sampling rate (Hz)
# n_fft: FFT length (e.g., 4 * fs)
wf = MPWhiteningFilter(psd, fs, n_fft)

# Get the one‑sided complex frequency response
# This response is minimum-phase (causal)
Wf = wf.frequency_response()
```

### 2. Computing Perturbative Corrections

If your template PSD (`psd2`) differs from the actual noise PSD (`psd1`), you can calculate the resulting geometric
drift corrections.

```python
import numpy as np
from zlw.corrections import PrtPsdDriftCorrection

# freqs:  One‑sided frequency grid
# psd1:   Realization/Noise PSD (The truth)
# psd2:   Template PSD (The model)
# htilde: Template waveform (frequency domain)

corr = PrtPsdDriftCorrection(
    freqs=freqs,
    psd1=psd1,
    psd2=psd2,
    htilde=htilde,
    fs=4096.0
)

# Calculate first-order geometric corrections
results = corr.correction()

print(f"Time Drift: {results.dt1:.4e} s")
print(f"SNR Loss:   {results.dsnr1:.4f}")
```

### Development

To install development dependencies and run the test suite (requires lalsuite):

```bash
# Install in editable mode with dev dependencies
pip install -e .[dev]

# Run tests
pytest
```
