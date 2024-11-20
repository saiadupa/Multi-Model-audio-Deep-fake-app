import os
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import requests
import yt_dlp as youtube_dl
from asd_models import ASD, ASD_GMM
from preprocessing import preprocess_audio
from pydub import AudioSegment

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        return output_path
    else:
        st.error("Error downloading file.")
        return None

def download_from_url(url, output_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_path, 'downloaded_audio.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            result = ydl.download([url])
            return os.path.join(output_path, 'downloaded_audio.wav')
        except Exception as e:
            st.error(f"Error downloading audio from URL. {str(e)}")
            return None

import ffmpeg

def extract_audio_from_video(video_file, output_audio_path):
    try:
        # Run ffmpeg command to extract audio
        ffmpeg.input(video_file).output(output_audio_path).run(quiet=True)
        return output_audio_path
    except ffmpeg.Error as e:
        st.error(f"FFmpeg error in extracting audio from video: {e.stderr.decode('utf8')}")
        return None

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sanitize_filename(filename):
    return re.sub(r'[^\w\-_\. ]', '_', filename)

def upload(data_file, model_selection):
    st.write(f"Uploaded file: {data_file.name}")

    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filename = sanitize_filename(data_file.name)
    speaker_name = os.path.splitext(filename)[0]

    file_path = os.path.join(data_dir, filename)

    try:
        with open(file_path, 'wb') as f:
            f.write(data_file.getbuffer())
    except Exception as e:
        st.error(f"Error saving uploaded file: {str(e)}")
        return

    st.success(f"File {filename} saved successfully!")

    audio_file = file_path

    if data_file.type.split('/')[0] == 'video':
        audio_file = os.path.join(data_dir, f"{speaker_name}.wav")
        audio_file = extract_audio_from_video(file_path, audio_file)
        if audio_file is None:
            return

    process_audio_file(audio_file, speaker_name, model_selection)

def convert_to_wav(audio_file):
    try:
        audio = AudioSegment.from_file(audio_file)
        wav_file = audio_file.replace(os.path.splitext(audio_file)[1], ".wav")
        audio.export(wav_file, format="wav")
        return wav_file
    except Exception as e:
        st.error(f"Error converting audio file to WAV format. {str(e)}")
        return None

def plot_audio_classification(audio_path, intervals, sr=None):
    audio, sr = librosa.load(audio_path, sr=sr)
    time = np.linspace(0, len(audio) / sr, len(audio))
    
    plt.figure(figsize=(10, 5))
    
    for label, start, end in intervals:
        start_sample, end_sample = int(start * sr), int(end * sr)
        if label == 'real':
            color = 'green'
        else:
            color = 'red'
        plt.plot(time[start_sample:end_sample], audio[start_sample:end_sample], color=color)
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Audio Classification: Real vs Deepfake")
    plt.legend(["Deepfake", "Real"], loc="lower left")
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

def process_audio_file(audio_file, speaker_name, model_selection):
    if not audio_file.endswith('.wav'):
        audio_file = convert_to_wav(audio_file)
        if audio_file is None:
            return

    try:
        y, sr = librosa.load(audio_file, sr=None)
        fig, ax = plt.subplots()
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
        ax.set(title='Mel-frequency spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error processing audio file. {str(e)}")
        return

    chunk_dir = f'chunks_{speaker_name}'
    try:
        audio_chunks_dict = preprocess_audio(audio_file, speaker_name=speaker_name, save_chunks=True, out_dir=chunk_dir)
    except Exception as e:
        st.error(f"Error preprocessing audio. {str(e)}")
        return

    model_scores = {}
    labeled_intervals = []

    try:
        for model_type in model_selection:
            if model_type in ['aasist', 'rawnet', 'rawgat']:
                asd_model = ASD(model_type=model_type, generate_score_file=True)
                score_df = asd_model.run(audio_chunks_dict, use_saved_chunks=True, chunk_dir=chunk_dir, speaker_name=speaker_name)
            elif model_type in ['cqcc', 'lfcc']:
                gmm_model = ASD_GMM(features=model_type, generate_score_file=True)
                score_df = gmm_model.produce_evaluation(audio_chunks_dict, speaker_name=speaker_name)
                
            def evaluate_scores(score_df):
                if 'start_time' not in score_df.columns:
                    score_df['start_time'] = score_df.index * 2.5  
                if 'end_time' not in score_df.columns:
                    score_df['end_time'] = score_df['start_time'] + 2.5 
                
                if 'cm-score' not in score_df.columns:
                    return score_df
                score_df['result'] = score_df['cm-score'].apply(lambda x: 'real' if x >= 0 else 'deepfake')
                return score_df

            score_df = evaluate_scores(score_df)
            model_scores[model_type] = score_df

    except Exception as e:
        st.error(f"Error evaluating models. {str(e)}")
        return

    try:
        cols = st.columns(len(model_selection))
        for i, model_type in enumerate(model_selection):
            with cols[i]:
                st.write(f"{model_type.upper()} Results:")
                st.dataframe(model_scores[model_type])

        if model_selection:
            avg_scores = pd.concat([model_scores[model]['cm-score'] for model in model_selection], axis=1).mean(axis=1)

            main_output_df = pd.DataFrame({
                'filename': model_scores[model_selection[0]]['filename'],
                'average_cm-score': avg_scores,
                'start_time': model_scores[model_selection[0]]['start_time'],
                'end_time': model_scores[model_selection[0]]['end_time'],
                'result': avg_scores.apply(lambda x: 'real' if x >= 0 else 'deepfake')
            })

            st.write("Main Output (Average of All Selected Models):")
            st.dataframe(main_output_df)

            labeled_intervals = [(row['result'], row['start_time'], row['end_time']) for index, row in main_output_df.iterrows()]

            avg_score_mean = main_output_df['average_cm-score'].mean()
            final_result = 'real' if avg_score_mean >= 0 else 'deepfake'
            avg_confidence = sigmoid(avg_score_mean) * 100

            if final_result == 'real':
                st.markdown(f"<p style='color:green;'>The audio file is classified as '{final_result.capitalize()}' with {avg_confidence:.2f}% confidence.</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color:red;'>The audio file is classified as '{final_result.capitalize()}' with {avg_confidence:.2f}% confidence.</p>", unsafe_allow_html=True)
    
        if labeled_intervals:
            plot_audio_classification(audio_file, labeled_intervals)
    
    except Exception as e:
        st.error(f"Error displaying results. {str(e)}")
        return

def main():
    st.title('Multi-model Audio Classification')
    st.write("Upload an audio or video file, or provide a URL to get started")

    st.sidebar.header('Select Models for Evaluation:')
    aasist_check = st.sidebar.checkbox('AASIST', value=True)
    rawnet_check = st.sidebar.checkbox('RawNet', value=True)
    rawgat_check = st.sidebar.checkbox('RawGAT', value=True)
    cqcc_check = st.sidebar.checkbox('CQCC (GMM)', value=True)
    lfcc_check = st.sidebar.checkbox('LFCC (GMM)', value=True)

    model_selection = []
    if aasist_check:
        model_selection.append('aasist')
    if rawnet_check:
        model_selection.append('rawnet')
    if rawgat_check:
        model_selection.append('rawgat')
    if cqcc_check:
        model_selection.append('cqcc')
    if lfcc_check:
        model_selection.append('lfcc')

    data_file = st.file_uploader("Upload Audio or Video File", type=["wav", "mp3", "m4a", "flac", "ogg", "mp4", "mkv", "avi", "mov"])
    
    url = st.text_input('Enter the URL of an audio or video file:')

    if data_file is not None:
        if 'video' in data_file.type:
            st.video(data_file)
        else:
            st.audio(data_file, format=f'audio/{data_file.type.split("/")[-1]}')
        upload(data_file, model_selection)
    elif url:
        filename = sanitize_filename(os.path.basename(url))
        speaker_name = os.path.splitext(filename)[0]
        
        file_path = os.path.join('./data', filename)
        
        audio_file = download_from_url(url, './data')
        if audio_file:
            st.success(f"Audio downloaded and extracted from URL: {url}")
            process_audio_file(audio_file, speaker_name, model_selection)

if __name__ == "__main__":
    main()