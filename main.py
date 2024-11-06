import os
import streamlit as st
import pandas as pd
from preprocessing import preprocess_audio
from asd_models import ASD

def upload(data_file):
    st.write(f"Uploaded file: {data_file.name}")

    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    filename = data_file.name
    speaker_name = os.path.splitext(filename)[0]  
    
    file_path = os.path.join(data_dir, filename)

    with open(file_path, 'wb') as f:
        f.write(data_file.getbuffer())

    st.success(f"File {filename} saved successfully!")

    audio_file = os.path.join(data_dir, filename)

    chunk_dir = f'chunks_{speaker_name}'
    audio_chunks_dict = preprocess_audio(audio_file, speaker_name=speaker_name,
                                         save_chunks=True, out_dir=chunk_dir)
    
    model_types = ['aasist', 'rawnet', 'rawgat']
    model_scores = {}

    for model_type in model_types:
        asd_model = ASD(model_type=model_type, generate_score_file=True)
        
        score_df = asd_model.run(audio_chunks_dict, use_saved_chunks=True, chunk_dir=chunk_dir, speaker_name=speaker_name)

        def evaluate_scores(score_df):
            if score_df is None or 'cm-score' not in score_df.columns:
                return score_df
            score_df['result'] = score_df['cm-score'].apply(lambda x: 'real' if x >= 0 else 'deepfake')
            return score_df
        
        score_df = evaluate_scores(score_df)
        model_scores[model_type] = score_df

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"Model AASIST Results:")
        st.dataframe(model_scores['aasist'])
    with col2:
        st.write(f"Model RAWNET Results:")
        st.dataframe(model_scores['rawnet'])
    with col3:
        st.write(f"Model RAWGAT Results:")
        st.dataframe(model_scores['rawgat'])

    avg_scores = pd.concat([model_scores['aasist']['cm-score'],
                            model_scores['rawnet']['cm-score'],
                            model_scores['rawgat']['cm-score']], axis=1).mean(axis=1)

    main_output_df = pd.DataFrame({
        'filename': model_scores['aasist']['filename'],
        'average_cm-score': avg_scores,
        'result': avg_scores.apply(lambda x: 'real' if x >= 0 else 'deepfake')
    })

    st.write("Main Output (Average of All Models):")
    st.dataframe(main_output_df)


def main():
    st.title('Multi model Audio Classification')
    st.write("Upload an audio file to get started")

    data_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a", "flac", "ogg"])

    if data_file is not None:
        st.audio(data_file, format=f'audio/{data_file.type.split("/")[-1]}')
        upload(data_file)

if __name__ == "__main__":
    main()