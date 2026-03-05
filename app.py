import os
import streamlit as st
import joblib
import pandas as pd

# Important to add src path for imports if needed, though clean_text could be redefined
import sys
base_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(base_dir, 'src')
sys.path.append(src_dir)

try:
    from preprocessing import clean_text
except ImportError:
    st.error("Failed to import preprocessing module. Ensure `src/preprocessing.py` exists.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    .prediction-true {
        color: #155724;
        background-color: #d4edda;
        border-color: #c3e6cb;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        margin-top: 20px;
    }
    .prediction-fake {
        color: #721c24;
        background-color: #f8d7da;
        border-color: #f5c6cb;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("📰 Fake News Detection Analyzer")
st.markdown("""
Welcome to the Fake News Detection tool! Paste the text of a news article below, 
and our Machine Learning model (Logistic Regression & TF-IDF) will predict if it's 
**Real News** or **Fake News**.
""")

# Load paths
model_path = os.path.join(base_dir, 'model', 'fake_news_model.pkl')
vectorizer_path = os.path.join(base_dir, 'model', 'tfidf_vectorizer.pkl')

@st.cache_resource
def load_assets(m_path, v_path):
    try:
        model = joblib.load(m_path)
        vectorizer = joblib.load(v_path)
        return model, vectorizer
    except Exception as e:
        return None, None

model, vectorizer = load_assets(model_path, vectorizer_path)

if not model or not vectorizer:
    st.warning("⚠️ Model or Vectorizer not found!")
    st.info("Please run `python src/train_model.py` to generate the `.pkl` files in the `model/` directory before using this app.")
    st.stop()

# Text input area
user_input = st.text_area("Paste Article Text Here:", height=250, placeholder="E.g., WASHINGTON (Reuters) - The president announced today...")

# Analyze button
if st.button("🔮 Analyze Article", type="primary"):
    if not user_input.strip():
        st.error("Please enter some text to analyze.")
    else:
        with st.spinner('Analyzing text...'):
            # 1. Clean Text
            cleaned_text = clean_text(user_input)
            
            # 2. Vectorize
            tfidf_features = vectorizer.transform([cleaned_text])
            
            # 3. Predict
            prediction = model.predict(tfidf_features)
            
            # 4. Display result
            # Assuming 1 = Real, 0 = Fake based on earlier training script
            if prediction[0] == 1:
                st.markdown('<div class="prediction-true">✅ Prediction: TRUE NEWS</div>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown('<div class="prediction-fake">🚨 Prediction: FAKE NEWS</div>', unsafe_allow_html=True)
                
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'><small>Built with Streamlit & Scikit-Learn | Portfolio Project</small></div>", unsafe_allow_html=True)
