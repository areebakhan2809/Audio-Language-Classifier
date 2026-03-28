import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ... (keep your existing imports and helper functions) ...

if audio_source is not None:
    st.audio(audio_source)
    
    if st.button("Analyze Audio", type="primary", use_container_width=True):
        # 1. LIVE ACTIVITY LOG
        with st.status("Model Activity: Processing Signal...", expanded=True) as status:
            st.write("📂 Creating temporary buffer...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tfile:
                tfile.write(audio_source.getbuffer())
                temp_path = tfile.name
            
            st.write("✂️ Resampling to 16kHz & Trimming to 5s...")
            st.write("🔊 Injecting Noise Regularization...")
            features = get_features(temp_path)
            
            st.write("🧠 Feeding 20 MFCCs to Random Forest...")
            prediction = model.predict(features.reshape(1, -1))[0]
            probabilities = model.predict_proba(features.reshape(1, -1))[0]
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # --- THE SHOWCASE SECTION ---
        st.markdown("### 🔍 Model Technical Insights")
        
        tab1, tab2 = st.tabs(["📊 Confidence Scores", "🧬 Acoustic Fingerprint"])
        
        with tab1:
            st.write("How certain was the AI about each language?")
            prob_df = pd.DataFrame({
                'Language': model.classes_,
                'Confidence': probabilities
            }).sort_values(by='Confidence', ascending=False)
            
            st.bar_chart(prob_df.set_index('Language'))
            st.success(f"🎯 **Final Decision: {prediction.upper()}**")
            st.balloons()

        with tab2:
            st.write("This is the 'Fingerprint' (MFCCs) the model actually 'saw':")
            fig, ax = plt.subplots(figsize=(10, 2))
            sns.heatmap(features.reshape(1, -1), annot=True, cmap="magma", cbar=False, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            st.pyplot(fig)
            st.caption("Lower coefficients represent the 'shape' of the vocal tract.")

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
