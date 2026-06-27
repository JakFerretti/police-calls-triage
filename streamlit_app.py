import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(page_title="Police Calls Triage AI", page_icon="🚨", layout="centered")

# --- 1. LOAD ARTIFACTS ---
@st.cache_resource
def load_models():
    model = tf.keras.models.load_model('cnn_police_model.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as handle:
        label_encoder = pickle.load(handle)
    return model, tokenizer, label_encoder

try:
    model, tokenizer, label_encoder = load_models()
except Exception as e:
    st.error("Error loading model files.")
    # This will display the actual traceback on the Streamlit dashboard for easier debugging
    st.exception(e) 
    st.stop()

# --- 2. USER INTERFACE ---
st.title("🚨 Police Emergency Calls Classification")
st.subheader("Deep Learning Triage System (1D CNN)")

user_input = st.text_area("Call Transcript:", placeholder="Type or paste text here...", height=150)

# --- 3. INFERENCE PIPELINE ---
if st.button("Classify Call", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # 1. CNN Preprocessing & Prediction
        sequences = tokenizer.texts_to_sequences([user_input])
        padded_data = pad_sequences(sequences, maxlen=100)
        
        with st.spinner("Neural network is analyzing text patterns..."):
            pred_probs = model.predict(padded_data)
            pred_class_idx = np.argmax(pred_probs, axis=1)[0]
            confidence = pred_probs[0][pred_class_idx] * 100
            
            # Retrieve the original class label (e.g., "Urgent", "Non-Urgent", "Prank")
            final_label = label_encoder.inverse_transform([pred_class_idx])[0]
        
        # 2. 🔥 SAFETY FILTER (Business Rule Guardrail)
        text_lower = user_input.lower()
        
        # Keywords associated with minor civic, traffic, or low-risk parking complaints
        downgrade_keywords = [
            'parked', 'parking', 'driveway', 'car blocking', 'illegally', 
            'bicycle', 'bike', 'sidewalk', 'abandoned bicycle', 'abandoned bike'
        ]

        # Override rule: if the CNN predicts 'Urgent' but the transcript clearly references
        # minor civic issues, always downgrade the priority to guarantee operational safety.
        if final_label.lower() == 'urgent':
            if any(keyword in text_lower for keyword in downgrade_keywords):
                # Force downgrade to Non-Urgent by matching the exact label encoder string format
                for c in label_encoder.classes_:
                    if c.lower() == 'non-urgent' or c.lower() == 'non_urgent':
                        final_label = c
                        break
                confidence = 100.0
                st.info("ℹ️ *Note: Prediction overridden by Civic & Traffic safety filter.*")

        # --- 4. CONDITIONAL VISUAL RESULTS ---
        st.markdown("---")
        st.subheader("Analysis Breakdown:")

        # Evaluate condition using lowercase to prevent any casing mismatches
        if final_label.lower() == 'urgent':
            st.error(f"🚨 **CLASS: {final_label.upper()}** (Confidence: {confidence:.2f}%)")
            st.toast("High priority call flagged instantly!", icon="🚨")
        elif final_label.lower() == 'prank':
            st.warning(f"🤡 **CLASS: {final_label.upper()}** (Confidence: {confidence:.2f}%)")
        else:
            st.info(f"ℹ️ **CLASS: {final_label.upper()}** (Confidence: {confidence:.2f}%)")
            
        # Class probabilities breakdown dropdown
        with st.expander("View CNN Probability Distribution Details"):
            for idx, class_name in enumerate(label_encoder.classes_):
                st.write(f"**{class_name}:** {pred_probs[0][idx]*100:.2f}%")
