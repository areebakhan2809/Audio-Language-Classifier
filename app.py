import streamlit as st
import librosa
import numpy as np
import joblib
import os

# --- 1. UI Configuration ---
st.set_page_config(page_title="Audio Language Classifier", page_icon="🎧", layout="centered")
st.title("🎧 Multilingual Audio Classifier")
st.write("Upload a 5-second audio clip, and the AI will predict if it is English, German, Spanish, or Hindi.")
st.write("Supported Formats: .wav, .flac, .ogg, or .mp3 .")
st.markdown("---")

# --- 2. Load the Model Securely ---
# @st.cache_resource ensures the model only loads once, keeping the app lightning fast
@st.cache_resource
def load_model():
    return joblib.load('lang_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error("❌ Error: Could not find 'lang_model.pkl'. Please ensure it is in the same folder as this script.")
    st.stop()

# --- 3. The Math Engine (Perfectly matched to Block 2) ---
def get_features(file_path):
    y, sr = librosa.load(file_path, sr=16000, duration=5.0)
    
    # White noise injection & 20 MFCCs to match training data shape
    noise_amplitude = 0.005 * np.random.uniform() * np.amax(y)
    y = y + noise_amplitude * np.random.normal(size=y.shape[0])
    mfcc_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    return np.mean(mfcc_data.T, axis=0)

# --- 4. The Frontend Upload Widget ---
uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'flac', 'ogg'])

if uploaded_file is not None:
    # Play the audio back to the user
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Predict Language", type="primary"):
        with st.spinner("Analyzing phonetic features..."):
            # Streamlit holds files in RAM. Librosa prefers hard drive paths.
            # We must write it to a temporary file, process it, and delete it.
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Extract and Predict
                features = get_features(temp_path).reshape(1, -1)
                prediction = model.predict(features)[0]
                
                # Display the victory
                st.success(f"🎯 **Predicted Language: {prediction.upper()}**")
                st.balloons() # Adds a nice presentation flair
                
            except Exception as e:
                st.error(f"Failed to process audio: {e}")
            finally:
                # ALWAYS clean up temporary files to prevent hard drive bloat
                if os.path.exists(temp_path):
                    os.remove(temp_path)
