import numpy as np
import matplotlib.pyplot as plt
import librosa

def plot_audio_classification(audio_path, fake_intervals, sr=None):
    """
    Plots an audio waveform with color-coded regions for classification.

    Parameters:
    - audio_path (str): Path to the audio file.
    - fake_intervals (list of tuples): List of (start_time, end_time) intervals in seconds marked as "fake".
    - sr (int, optional): Sampling rate for the audio file (if None, it is inferred).
    """
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=sr)
    
    # Convert fake intervals to sample indices
    fake_samples = [(int(start * sr), int(end * sr)) for start, end in fake_intervals]
    
    # Create a time axis
    time = np.linspace(0, len(audio) / sr, len(audio))
    
    # Plot the waveform
    plt.figure(figsize=(10, 5))
    
    # Traverse the waveform and color-code the sections
    last_idx = 0
    for start, end in fake_samples:
        # Plot the "real" section before the current "fake" section
        plt.plot(time[last_idx:start], audio[last_idx:start], color="green")
        # Plot the "fake" section
        plt.plot(time[start:end], audio[start:end], color="red")
        last_idx = end  # Update the last index
    
    # Plot the remaining "real" section
    if last_idx < len(audio):
        plt.plot(time[last_idx:], audio[last_idx:], color="green")
    
    # Add labels, legend, and title
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Audio Classification: Real vs Deepfake")
    plt.legend(["Real (Green)", "Fake (Red)"], loc="lower left")
    plt.tight_layout()
    
    # Show the plot
    plt.show()

# Specify the audio file and fake intervals
audio_file = "knownVocals.wav"  # Replace with your actual file path
fake_intervals = [(0, 10), (40, 50)]  # Intervals in seconds

# Call the function
plot_audio_classification(audio_file, fake_intervals)
