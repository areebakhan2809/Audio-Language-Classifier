import streamlit as st
import librosa
import numpy as np
import joblib
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Audio Language Classifier", page_icon="🎧", layout="centered")

st.title("🎧 Multilingual Audio Classifier")
st.markdown("""
This AI model identifies the spoken language from a 5-second audio clip.
**Languages Supported:** English, German, Spanish, Hindi.
""")
st.markdown("---")

# 1. Load the Model (Cached for Speed)
@st.cache_resource
def load_model():
    return joblib.load('lang_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error("❌ Error: Could not find 'lang_model.pkl'. Please ensure it is in the repository.")
    st.stop()

# 2. Robust Feature Extraction
def get_features(file_path):
    # Load 5 seconds at 16kHz
    y, sr = librosa.load(file_path, sr=16000, duration=5.0)
    
    # Noise Regularization (Forces model to look at phonetic shapes)
    noise_amplitude = 0.005 * np.random.uniform() * np.amax(y)
    y = y + noise_amplitude * np.random.normal(size=y.shape[0])
    
    # Extract 20 MFCCs
    mfcc_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc_data.T, axis=0)

# 3. UI: File Upload
uploaded_file = st.file_uploader("📂 Upload a 5-second audio file", type=['wav', 'mp3', 'flac', 'ogg'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("🚀 Analyze & Predict", type="primary", use_container_width=True):
        
        # --- MODEL THINKING: Step-by-Step Logs ---
        with st.status("Model Activity: Processing Signal...", expanded=True) as status:
            st.write("📁 Creating temporary data buffer...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tfile:
                tfile.write(uploaded_file.getbuffer())
                temp_path = tfile.name
            
            st.write("✂️ Normalizing temporal duration to 5.0s...")
            st.write("🔊 Applying White Noise Regularization...")
            features = get_features(temp_path)
            
            st.write("🧠 Feeding MFCC vectors to Random Forest...")
            prediction = model.predict(features.reshape(1, -1))[0]
            probabilities = model.predict_proba(features.reshape(1, -1))[0]
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # --- THE TECHNICAL DASHBOARD ---
        st.markdown("### 🔍 Model Diagnostic Insights")
        
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Confidence Scores**")
            # Create a clean dataframe for the bar chart
            prob_df = pd.DataFrame({
                'Language': model.classes_,
                'Probability': probabilities
            }).sort_values(by='Probability', ascending=False)
            
            st.bar_chart(prob_df.set_index('Language'))

        with col2:
            st.markdown("**Acoustic Fingerprint**")
            # Visualize the 20 MFCC values as a heatmap
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(features.reshape(-1, 1), annot=True, cmap="magma", cbar=False, ax=ax)
            ax.set_title("20 MFCC Coefficients")
            ax.set_xticks([])
            ax.set_ylabel("Coefficient Index")
            st.pyplot(fig)

        # FINAL RESULT
        st.markdown("---")
        st.subheader(f"🎯 Predicted Language: :green[{prediction.upper()}]")
        st.balloons()

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Footer for B.Tech context
st.markdown("---")
st.caption("Developed by Mamu | B.Tech CSE | Nagpur, India")
