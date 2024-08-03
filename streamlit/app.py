import streamlit as st
import os
from src.pipeline.predict_pipeline import PredictPipeline

pred_obj = PredictPipeline()

st.title('Audio Classification')

st.write('Upload audio and click "Predict".')
audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if audio_file is not None:
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    audio_path = os.path.join(temp_dir, audio_file.name)
    
    with open(audio_path, "wb") as f:
        f.write(audio_file.getbuffer())

    st.audio(audio_file)

    if st.button("Predict"):
        prediction = pred_obj.predict(audio_path)
        st.write(f"Predicted Class: {prediction}")

        os.remove(audio_path)
        st.write("Temporary file deleted.")