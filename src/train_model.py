import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Import the custom preprocessing function
from preprocessing import clean_text

def train_and_save_model():
    """
    Trains a Fake News Detection model using TF-IDF and Logistic Regression.
    Saves the trained model and vectorizer to the 'model/' directory.
    """
    print("🚀 Starting Model Training Process...")

    # 1. Define Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fake_path = os.path.join(base_dir, 'dataset', 'Fake.csv')
    true_path = os.path.join(base_dir, 'dataset', 'True.csv')
    model_dir = os.path.join(base_dir, 'model')

    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # 2. Load Data
    try:
        print("Loading datasets...")
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure Fake.csv and True.csv exist in the dataset/ directory.")
        return

    # 3. Add Labels (0 for Fake, 1 for Real)
    fake_df['label'] = 0
    true_df['label'] = 1

    # 4. Merge Datasets
    # We only really need the 'text' column for training, but you can also combine 'title' + 'text'
    # For simplicity, we'll just use the 'text' column.
    df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
    
    # Check if necessary columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
        print("❌ Error: Datasets must contain 'text' column.")
        return

    # Drop any missing values
    df = df.dropna(subset=['text'])

    # 5. Preprocess Text
    print("🧹 Cleaning text data (this may take a while on large datasets)...")
    # Using pandas apply to clean the text column
    df['cleaned_text'] = df['text'].apply(clean_text)

    # 6. Split Data
    print("🔀 Splitting data into train and test sets...")
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 7. Feature Extraction (TF-IDF Vectorization)
    print("🔢 Applying TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(max_features=5000) # Limit features to top 5000 words
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 8. Model Training
    print("🧠 Training Logistic Regression model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train_tfidf, y_train)

    # 9. Model Evaluation
    print("📊 Evaluating model on test data...")
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake (0)', 'Real (1)']))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 10. Save Model and Vectorizer
    model_path = os.path.join(model_dir, 'fake_news_model.pkl')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    
    print("\n💾 Saving model and vectorizer...")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    print("🎉 Training process completed successfully!")

if __name__ == "__main__":
    train_and_save_model()
