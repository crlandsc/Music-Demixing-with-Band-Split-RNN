import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io.wavfile as wav
from tqdm import tqdm
from scipy.signal import butter, filtfilt, freqz

def apply_low_pass_filter(signal, cutoff_freq, order, fs):
    """
    Apply a low-pass filter to a stereo audio signal.

    Args:
        signal (numpy.ndarray): Stereo audio signal with shape (2, N), where N is the number of samples.
        cutoff_freq (float): Cutoff frequency of the low-pass filter in Hz.
        fs (float): Sampling rate of the audio signal in Hz.

    Returns:
        numpy.ndarray: Filtered stereo audio signal with shape (2, N).
    """
    # Design the low-pass filter
    nyquist_freq = 0.5 * fs
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    b, a = butter(order, normalized_cutoff_freq, btype='low', analog=False, output='ba')

    # Determine the padding length
    padlen = max(3 * max(len(a), len(b)), signal.shape[1])

    # Apply the filter to the left and right channels separately
    filtered_signal = np.zeros_like(signal)
    for i in range(2):
        filtered_signal[:, i] = filtfilt(b, a, signal[:, i])#, padlen=padlen)

    return filtered_signal

def plot_low_pass_filter_response(cutoff_freq, order, fs):
    """
    Plot the frequency response of a low-pass filter.

    Args:
        cutoff_freq (float): Cutoff frequency of the low-pass filter in Hz.
        fs (float): Sampling rate of the filter in Hz.
    """
    # Design the low-pass filter
    nyquist_freq = 0.5 * fs
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    b, a = butter(order, normalized_cutoff_freq, btype='low', analog=False, output='ba')

    # Calculate the frequency response of the filter
    w, h = freqz(b, a, fs=fs)

    # Convert amplitude response to dB
    amplitude_db = 20 * np.log10(np.abs(h))

    # Plot the frequency response
    plt.figure()
    plt.plot(w, amplitude_db)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Low-Pass Filter Frequency Response')
    plt.grid(True)
    plt.xscale('log')
    plt.xticks([31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
               [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    plt.ylim([-60, 5])
    plt.show()


path = "I:/MDX-23/A_Label_Noise_Processed/"
folder = "Bass/train/fa46f72c-696d-45bc-bcc5-2b3305800565"
bass_file = f"{path}{folder}/bass.wav"
output_file = f"{path}{folder}/bass_.wav"
cutoff = 600 # 250
order = 2 # 5

# Load audio file
sr, input_signal = wav.read(os.path.abspath(bass_file)) # white_noise

# Filter audio
filtered_signal = apply_low_pass_filter(input_signal, cutoff, order, sr)

# Filter audio (x2)
filtered_signal = apply_low_pass_filter(filtered_signal, 2000, 8, sr)

# Save the combined data as a WAV file
wav.write(output_file, sr, filtered_signal)
print("Output WAV file saved as:", output_file)

# Plot response
# plot_low_pass_filter_response(cutoff, order, sr)

