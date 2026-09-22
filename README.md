# Evolutionary Inverse Design and Reservoir Computing with Spin Waves

Micromagnetic simulation of Yttrium Iron Garnet (YIG) nanostructures for two problems in magnonic computing: **genetic-algorithm inverse design of a frequency demultiplexer**, and **physical reservoir computing for spoken digit classification**. Both components are built on [MuMax3](https://mumax.github.io/).

> BSc Computer Science dissertation, University of Manchester (2026). Supervised by William Griggs.

## Overview

Magnonics — computing with spin waves, the collective oscillations of electron spins in a magnetic material — is a candidate low-power alternative to conventional electronics. This project explores two ways of putting that to use on the same simulated YIG platform:

1. **Inverse design** — a genetic algorithm searches for the physical arrangement of defects that produces a target device behaviour, rather than deriving it analytically.
2. **Reservoir computing** — a fixed, complex physical system transforms an input signal into a high-dimensional response, which a simple trained linear readout then decodes.

## What's in here

**1. Demultiplexer inverse design.** A GA optimises the positions of 14 YIG defect pillars in a waveguide to route 2.6 GHz and 2.8 GHz spin waves into two separate output channels, evolving against a Direct Binary Search baseline inspired by Wang et al.

**2. Reservoir computing for spoken digit recognition.** The same simulation infrastructure drives a YIG-disk reservoir with audio envelopes from the Free Spoken Digit Dataset (FSDD); a Ridge classifier reads out the resulting magnetisation response.

## Results

- **Demultiplexer:** best GA configuration reached a fitness of **1.04 × 10⁻²**, ~2 orders of magnitude above the random baseline, with routing ratios of **3.93:1** (2.6 GHz) and **2.81:1** (2.8 GHz). A Direct Binary Search baseline (~300 evaluations) failed to reach this in a comparable budget.
- **Reservoir:** a Ridge classifier reached **15.4% accuracy** on the 10-class FSDD task (chance = 10%). Feature-space analysis (PCA, variance ratio) traced the ceiling to insufficient nonlinear class separation in the fixed reservoir geometry — a diagnosed negative result, not an unexplained one.

## How it works (short version)

- Both devices are planar YIG structures simulated cell-by-cell in MuMax3, with a bias field to establish the spin wave regime and absorbing boundaries to suppress reflections.
- The demultiplexer's fitness is a **multiplicative** contrast between the two output detectors at their target frequencies — this was chosen after an additive version collapsed both frequencies onto a single output.
- The reservoir pipeline extracts an audio envelope via the Hilbert transform, injects it into MuMax3 as a long sequence of field updates (MuMax3 can't read external waveforms at runtime), and turns the resulting magnetisation dynamics at three detectors into a 450-dimensional feature vector per sample.

Full derivations, parameters, and figures are in the dissertation PDF.

## Requirements

- [MuMax3](https://mumax.github.io/) (GPU-accelerated micromagnetic simulation)
- Python 3 with `numpy`, `scipy`, `scikit-learn`, `matplotlib`
- An NVIDIA GPU is strongly recommended — simulations take minutes each and the full dataset run took ~60–70 hours of compute.

## Limitations & future work

Results are from single random seeds and one fixed, unoptimised reservoir geometry, so they should be read as a proof of concept rather than a tuned system. The most promising next step is applying the same GA framework to reservoir geometry itself, plus richer audio encodings that preserve phase information the current envelope-based approach discards. Details in the dissertation.

## Acknowledgements

Supervised by William Griggs, Department of Computer Science, University of Manchester.

## License

Add a license of your choice here (e.g. MIT) if you intend the code to be reused.
