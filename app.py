import streamlit as st
import librosa
import numpy as np
import joblib
import os
import tempfile 

st.set_page_config(page_title="Audio Language Classifier", page_icon="🎧", layout="centered")
st.title("🎧 Multilingual Audio Classifier")
st.write("Upload an audio clip or record your voice live! The AI will predict if it is English, German, Spanish, or Hindi.")
st.markdown("---")

@st.cache_resource
def load_model():
    return joblib.load('lang_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error("❌ Error: Could not find 'lang_model.pkl'. Please ensure it is in the repository.")
    st.stop()

def get_features(file_path):
    y, sr = librosa.load(file_path, sr=16000, duration=5.0)
    noise_amplitude = 0.005 * np.random.uniform() * np.amax(y)
    y = y + noise_amplitude * np.random.normal(size=y.shape[0])
    mfcc_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc_data.T, axis=0)

# --- THE NEW UI: Side-by-Side Inputs ---
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📂 Upload an audio file", type=['wav', 'mp3', 'flac', 'ogg'])

with col2:
    # This single line generates a fully functional microphone recording widget!
    recorded_audio = st.audio_input("🎙️ Or record your voice")

# Logic to choose which audio source to process
audio_source = recorded_audio if recorded_audio else uploaded_file

if audio_source is not None:
    # Playback the selected audio
    st.audio(audio_source)
    
    if st.button("Predict Language", type="primary", use_container_width=True):
        with st.spinner("Analyzing phonetic features..."):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tfile:
                tfile.write(audio_source.getbuffer())
                temp_path = tfile.name
            
            try:
                features = get_features(temp_path).reshape(1, -1)
                prediction = model.predict(features)[0]
                
                st.success(f"🎯 **Predicted Language: {prediction.upper()}**")
                st.balloons() 
                
            except Exception as e:
                st.error(f"Failed to process audio: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
