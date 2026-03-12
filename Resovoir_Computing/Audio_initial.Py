import numpy as np
import soundfile as sf
from scipy.signal import hilbert, resample
import matplotlib.pyplot as plt
import os

# =========================
# SETTINGS
# =========================
# Set your file path, how much to shrink the data, and max field strength
wav_file = "Resovoir_Computing/free-spoken-digit-dataset/recordings/0_george_21.wav"   # change to your file
downsample_factor = 5          # increase to reduce simulation length
B_scale = 0.01                 # 0.01 Tesla = 10 mT max field
output_file = "input_signal.txt"

# =========================
# LOAD AUDIO
# =========================
def create_envelope(wav_file, downsample_factor, B_scale, output_file):
    data, sr = sf.read(wav_file)

    # If stereo, convert to mono
    # We only want a single audio track to drive our 1D magnetic field
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    print(f"Sample rate: {sr} Hz")
    print(f"Original samples: {len(data)}")
    print(f"Duration: {len(data)/sr:.3f} seconds")

    # NORMALISE WAVEFORM
    data = data / np.max(np.abs(data))

    # EXTRACT ENVELOPE
    # Get the smooth outer shape of the sound wave. 
    analytic_signal = hilbert(data)
    envelope = np.abs(analytic_signal)

    # DOWNSAMPLE
    # this forces every audio signal to have exactly 4000 time steps
    target_points = 4000
    envelope = resample(envelope, target_points)

    print(f"Samples after resampling: {len(envelope)}")

    # NORMALISE AGAIN
    envelope = envelope / np.max(envelope)

    # SCALE TO TESLA
    # Multiply the 0-to-1 envelope by my chosen max Tesla value (0.01)
    signal = B_scale * envelope

    # SAVE FOR MUMAX3
    # Export the final numbers to a text file that MuMax3 can read as an input
    np.savetxt(output_file, signal)

    print(f"Signal saved to {output_file}")
    print(f"Max field amplitude: {np.max(signal):.4f} Tesla")

    #plot
    plt.figure()
    plt.plot(signal)
    plt.title("Signal Injected into MuMax3 (Tesla)")
    plt.xlabel("Time Step")
    plt.ylabel("Magnetic Field (T)")
    plt.show()
