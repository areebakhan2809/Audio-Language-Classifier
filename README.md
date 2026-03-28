# 🎧 Multilingual Audio Language Classifier

An end-to-end Machine Learning application that identifies the spoken language from a 5-second audio clip. Built with Python, Scikit-Learn, and Streamlit, this project transition from a Google Colab research environment to a live, cloud-hosted production app.

## 🚀 Live Demo
Check out the live application here: [https://audio-language-classifier.streamlit.app/](https://audio-language-classifier.streamlit.app/)

## 🌟 Key Features
* **Dual-Input System:** Supports both `.wav/.mp3` file uploads and **live microphone recording** directly in the browser.
* **Signal Processing:** Utilizes **MFCC (Mel-frequency cepstral coefficients)** for high-accuracy phonetic feature extraction.
* **Robustness Engineering:** Implemented **Synthetic Data Augmentation** (White Noise Injection and Pitch Shifting) to handle real-world room acoustics.
* **Cloud Native:** Fully deployed via Streamlit Community Cloud with a GitHub-integrated CI/CD pipeline.

## 🛠️ Technical Stack
* **Language:** Python 3.12
* **AI/ML:** Scikit-Learn (Random Forest Classifier), Joblib
* **Audio Analysis:** Librosa, NumPy
* **Frontend:** Streamlit
* **Data Source:** Mozilla Common Voice (Verified Subset via Kaggle)

## 📊 How It Works
1.  **Preprocessing:** Audio is resampled to 16kHz and trimmed/padded to exactly 5.0 seconds.
2.  **Feature Extraction:** The system calculates the mean of 20 MFCCs across the time domain, creating a unique mathematical "fingerprint" for the audio.
3.  **Classification:** A Random Forest model analyzes the fingerprint to predict one of four languages: **English, Hindi, Spanish, or German**.
4.  **Deployment:** The model is "pickled" and served via a Streamlit web interface.

## 📁 Project Structure
```text
.
├── app.py               # Streamlit web application & UI logic
├── lang_model.pkl       # Pre-trained Random Forest model
├── requirements.txt     # Production dependencies (Numpy 1.26.4 for stability)
└── README.md            # Project documentation
