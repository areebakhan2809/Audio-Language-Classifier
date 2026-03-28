# 🎧 Multilingual Audio Language Classifier

An end-to-end MLOps pipeline designed to classify spoken audio into four distinct languages: **English, German, Spanish, and Hindi**. 

This project demonstrates a complete software engineering workflow—from automated dataset ingestion and advanced signal processing to a live, cloud-hosted production app.

## 🚀 Live Demo
**Access the App:** [https://audio-language-classifier.streamlit.app/](https://audio-language-classifier.streamlit.app/)

## 🌟 Key Features
* **Robust Signal Processing:** Utilizes **MFCC (Mel-frequency cepstral coefficients)** for high-accuracy phonetic fingerprinting.
* **Noise-Hardened Training:** Implements **Synthetic Data Augmentation** (White Noise Injection and Pitch Shifting) to bridge the "Domain Gap" between clean datasets and real-world room acoustics.
* **Production Deployment:** Fully integrated with Streamlit Community Cloud and GitHub CI/CD.

## 🧠 Model Architecture & Engineering
The pipeline relies on `librosa` for audio processing and a `RandomForestClassifier` for categorization. To ensure the model learns linguistic phonetics rather than dataset-specific artifacts, the feature extraction layer implements three strict regularizations:

1.  **Temporal Cropping:** Audio is strictly cropped to 5.0 seconds at 16kHz to maximize information density.
2.  **White Noise Injection:** A microscopic layer of dynamic static is applied to the raw waveform to mask compression artifacts and force the model to rely on gross phonetic shapes.
3.  **Dimensionality Reduction:** Extracts exactly 20 MFCCs. Dropping upper bands (21-40) effectively blinds the AI to recording hardware bias and room echoes.

**Performance:** Achieves a robust **74% overall accuracy** on a crowdsourced, multi-speaker dataset, representing true "Language ID" generalization.

## 📊 The Dataset
This pipeline utilizes a verified and balanced audio dataset published on Kaggle, engineered to prevent "Voice ID" bias.

* **Dataset Link:** [Robust 4-Language Audio Dataset (4,000 Files)](https://www.kaggle.com/datasets/gamesnnukes/robust-4-language-audio-dataset-4000-files)
* **Volume:** 4,000 files (1,000 per language class).
* **Sources:** Features diverse conversational speech (including Mozilla Common Voice for Hindi).

## 🛠️ Technical Stack
* **Language:** Python 3.12.13
* **AI/ML:** Scikit-Learn, Joblib
* **Audio Analysis:** Librosa, NumPy
* **Frontend:** Streamlit
* **Infrastructure:** GitHub, Streamlit Cloud

## 🏗️ Local Setup
1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/YourUsername/Audio-Language-Classifier.git](https://github.com/YourUsername/Audio-Language-Classifier.git)
    cd Audio-Language-Classifier
    ```
2.  **Install Dependencies:**
    ```bash
    python -m pip install -r requirements.txt
    ```
3.  **Run the App:**
    ```bash
    python -m streamlit run app.py
    ```

## 📂 Project Structure
```text
.
├── app.py               # Streamlit UI & Inference logic
├── lang_model.pkl       # Pre-trained Random Forest model
├── requirements.txt     # Production dependencies (Numpy 1.26.4)
└── README.md            # Project documentation
