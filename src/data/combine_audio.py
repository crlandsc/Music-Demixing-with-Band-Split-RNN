import os
import numpy as np
import scipy.io.wavfile as wav

def combine_wav_files(file1, file2, output_file):
    """
    Combines two WAV files by simple addition and saves the output WAV file.

    Args:
        file1 (str): Path to the first WAV file.
        file2 (str): Path to the second WAV file.
        output_file (str): Path to save the output WAV file.
    """
    # Use os.path.join to create absolute paths
    file1 = os.path.abspath(file1)
    file2 = os.path.abspath(file2)
    output_file = os.path.abspath(output_file)

    # Load the input WAV files
    rate1, data1 = wav.read(file1)
    rate2, data2 = wav.read(file2)

    # Make sure the sample rates of the input WAV files are the same
    if rate1 != rate2:
        raise ValueError("Sample rates of the input WAV files do not match")

    # Make sure the number of channels and sample widths of the input WAV files are the same
    if data1.shape != data2.shape:
        raise ValueError("Number of channels and sample widths of the input WAV files do not match")

    # Combine the WAV files by simple addition
    combined_data = data1 + data2

    # Save the combined data as a WAV file
    wav.write(output_file, rate1, combined_data)

    print("Output WAV file saved as:", output_file)


# Specify the paths to the input WAV files and the output file
path = "I:/MDX-23/A_Label_Noise_Processed/"
folder = "Bass/train/0f5fb60c-51d4-4618-871d-650c9e927b79"
file1 = f"{path}{folder}/bass.wav"
file2 = f"{path}{folder}/bass_lp.wav"
output_file = f"{path}{folder}/output.wav"

# Call the combine_wav_files() function
combine_wav_files(file1, file2, output_file)
