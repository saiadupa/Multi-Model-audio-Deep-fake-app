import os
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def trim_silence(audio, silence_thresh=-40, min_silence_len=500, buffer=500):
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    
    if not nonsilent_ranges:
        print(f"No nonsilent sections detected, audio file may be entirely silent.")
        return None
    
    start_trim = max(0, nonsilent_ranges[0][0] - buffer)
    end_trim = min(len(audio), nonsilent_ranges[-1][1] + buffer)
    
    trimmed_audio = audio[start_trim:end_trim]
    return trimmed_audio

def repeat_audio(audio, target_duration_ms):
    while len(audio) < target_duration_ms:
        audio += audio
    return audio[:target_duration_ms]

def preprocess_audio(audio_file, speaker_name='unknown', chunk_duration=12, format=None, save_chunks=False, out_dir='./chunks/'):
    if save_chunks:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        elif len(os.listdir(out_dir)) != 0:
            files = os.listdir(out_dir)
            for f in files:
                os.remove(os.path.join(out_dir, f))

    if format is None:
        _, f_ext = os.path.splitext(audio_file)
        format = f_ext[1:].lower()

    audio = AudioSegment.from_file(audio_file, format=format).set_channels(1)    
    audio = trim_silence(audio)
    if audio is None:
        return {}

    total_duration = len(audio)
    
    # Ensure the audio length is at least as long as the chunk duration
    target_duration_ms = chunk_duration * 1000  # 12 seconds in milliseconds
    audio = repeat_audio(audio, target_duration_ms)
    
    base_index = 0
    chunk_index = 1
    start_time = 0

    audio_dict = {}

    while start_time < total_duration:
        end_time = start_time + target_duration_ms
        chunk = audio[start_time:end_time]

        chunk = repeat_audio(chunk, target_duration_ms)

        if not (is_silent(chunk) or len(chunk) < 5000):
            if save_chunks:
                out_filename = f"{speaker_name}_{base_index + chunk_index}.flac"
                output_file = os.path.join(out_dir, out_filename)
                chunk.export(output_file, format="flac")
            
            chunk_np = np.array(chunk.get_array_of_samples(), dtype=np.float32)
            audio_dict[out_filename] = chunk_np

        start_time += target_duration_ms
        chunk_index += 1

    return audio_dict

def is_silent(audio_segment):
    rms = audio_segment.rms
    silence_threshold = 0  
    return rms < silence_threshold
