import pandas as pd
import numpy as np
from importlib import import_module
from typing import Dict, List, Union
from tqdm import tqdm
import os
import glob
from pydub import AudioSegment

def preprocess_audio(audio_file, speaker_name='unknown', chunk_duration=10, format=None, save_chunks=False, out_dir='./chunks/'):

    print(audio_file)
    print(speaker_name)
    
    # create output directory. If output dircetory exists, remove all contents of the output directory
    if save_chunks:
        if not os.path.exists(out_dir): 
            os.makedirs(out_dir)
        elif len(os.listdir(out_dir)) != 0:             
            files = os.listdir(out_dir)
            print(files)
            for f in files:
                os.remove(os.path.join(out_dir, f))

    # Determine the format if not explicitly provided
    if format is None:
        _, f_ext = os.path.splitext(audio_file)
        format = f_ext[1:].lower()

    # read audio file
    audio = AudioSegment.from_file(audio_file, format=format).set_channels(1)
    
    sr = audio.frame_rate

    # total duration of the audio file in milliseconds
    total_duration = audio.duration_seconds * 1000

    print(total_duration)

    base_index = 0
    chunk_index = 1
    start_time = 0
    end_time = 0
    step_size = 5000

    audio_dict = {}

    while end_time <= total_duration:
        
        end_time = start_time + chunk_duration * 1000
        
        file_index = base_index + chunk_index

        out_filename = speaker_name + f"_{file_index}"

        chunk = audio[start_time:end_time]

        if not (is_silent(chunk) or len(chunk) < 5000):

            if save_chunks:
                output_file = os.path.join(out_dir, out_filename + '.flac')
                chunk.export(output_file, format="flac")
            
            chunk_np = np.array(chunk.get_array_of_samples(), dtype=np.float32)
        
            audio_dict[out_filename] = chunk_np

        start_time = start_time + step_size
        chunk_index += 1

    return audio_dict


def is_silent(audio_segment):
    """
    Checks if an audio segment is just silence.

    Args:
    audio_segment: A pydub.AudioSegment object.

    Returns:
    True if the audio segment is just silence, False otherwise.
    """
    rms = audio_segment.rms
    silence_threshold = 0  # dB
    return rms < silence_threshold