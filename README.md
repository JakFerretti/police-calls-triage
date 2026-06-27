### Project Overview
In emergency response management, every second counts. This notebook presents an end-to-end Machine Learning and Deep Learning pipeline designed to automatically triage incoming police emergency call transcripts. The system acts as an AI dispatch assistant, instantly categorizing transcripts into one of three operational classes:
* **🚨 URGENT:** High-priority, life-threatening situations requiring immediate tactical dispatch (e.g., crimes in progress, severe accidents).
* **ℹ️ NON-URGENT:** Standard civic, administrative, or non-hazardous traffic complaints handled via routine response channels.
* **🤡 PRANK:** Mocking or absurd calls designed to disrupt emergency lines, which can be safely filtered or flagged for manual review.

---

### Key Pipeline Stages
1. **Exploratory Data Analysis (EDA):** Text length analysis, class distribution tracking, and token profiling.
2. **Baseline Development:** Implementation of a statistical **TF-IDF Vectorizer + LightGBM** model to set a strong benchmark.
3. **Deep Learning Framework:** Construction of a **1D Convolutional Neural Network (CNN)** equipped with an embedding layer to capture sequence layout and contextual word patterns.
4. **Production Engineering:** Integration of a deterministic **Civic & Traffic Safety Guardrail** to eliminate statistical shortcut learning.
5. **Interactive Deployment:** A complete script export to power a local **Streamlit** dashboard for real-time model interaction.

---

# 🚀 Live Demo
You can test the interactive deployment of this model here: 
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/JackLittleIron/police-triage-app)

---

# 🔬 Model Training & Workflow
The complete training pipeline, text cleaning with Polars, and 1D-CNN optimization can be viewed directly here:
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/jacopoferretti/police-call-triage-ai-1d-cnn-streamlit-app)
