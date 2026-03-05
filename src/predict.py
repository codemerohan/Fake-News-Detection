import os
import joblib

# Import custom text cleaning logic
from preprocessing import clean_text

def predict_news(text, model_path, vectorizer_path):
    """
    Given an input string of text, loads the trained model and vectorizer
    to predict if the news is Fake or True.
    
    Args:
        text (str): The raw text of the news article.
        model_path (str): The path to the saved model (.pkl).
        vectorizer_path (str): The path to the saved vectorizer (.pkl).
        
    Returns:
        str: "True News" or "Fake News"
    """
    # 1. Input Validation
    if not isinstance(text, str) or not text.strip():
        return "Invalid input text."
        
    # 2. Check Paths
    if not os.path.exists(model_path):
        return f"Error: Model not found at {model_path}"
    if not os.path.exists(vectorizer_path):
        return f"Error: Vectorizer not found at {vectorizer_path}"

    try:
        # 3. Load Model and Vectorizer
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # 4. Clean and Vectorize Input Text
        cleaned_text = clean_text(text)
        tfidf_features = vectorizer.transform([cleaned_text])
        
        # 5. Make Prediction
        prediction = model.predict(tfidf_features)
        
        # 6. Format Result (0 = Fake, 1 = Real based on our training labels)
        if prediction[0] == 1:
            return "True News"
        else:
            return "Fake News"
            
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m_path = os.path.join(base_dir, 'model', 'fake_news_model.pkl')
    v_path = os.path.join(base_dir, 'model', 'tfidf_vectorizer.pkl')

    if len(sys.argv) > 1:
        input_text = sys.argv[1]
        result = predict_news(input_text, m_path, v_path)
        print(f"Prediction for input '{input_text[:50]}...':")
        print(f"==> {result}")
    else:
        print("Usage: python predict.py \"Your news text here.\"")
