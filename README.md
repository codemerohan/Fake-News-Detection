# 📰 Fake News Detection System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)](https://scikit-learn.org/)

A comprehensive Machine Learning project to classify news articles as **Fake** or **True** using Natural Language Processing (NLP) techniques and Logistic Regression. 

This project demonstrates an end-to-end Machine Learning pipeline: Data Collection, Exploratory Data Analysis (EDA), Text Preprocessing (NLTK), Feature Engineering (TF-IDF), Model Training, and Web Application Deployment via Streamlit.

---

## 🌟 Key Features
- **Accurate Classification**: Uses a Logistic Regression classifier trained on a robust TF-IDF vectorized dataset.
- **NLP Preprocessing**: Custom text pipeline removing noise (URLs, HTML, punctuation), handling stopwords, and applying Porter Stemming.
- **Interactive Web UI**: A clean, responsive user interface built with Streamlit allowing users to test articles in real-time.
- **In-depth EDA**: Jupyter Notebook provided encompassing data visualization and Word Clouds.

## 📂 Project Structure

```text
Fake-News-Detection/
│
├── dataset/                    # Directory for raw CSV data (Fake.csv, True.csv)
│   └── README.md               # Instructions on getting the ISOT Dataset
├── model/                      # Stores serialized ML models and vectorizers (.pkl)
├── notebooks/                  
│   └── fake_news_eda.ipynb     # Jupyter Notebook for EDA & Model experimentation
├── src/
│   ├── preprocessing.py        # Core NLP text cleaning logic
│   ├── train_model.py          # Script to execute the ML training pipeline
│   └── predict.py              # CLI utility for quick predictions
├── app.py                      # Main Streamlit Web Application
├── requirements.txt            # Python dependencies
└── README.md                   # Project Documentation
```

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Download the Dataset
The `dataset/` folder initially contains only dummy samples to maintain a lightweight repository.
To train a highly accurate model, download the **ISOT Fake News Dataset** from Kaggle:
* See [dataset/README.md](dataset/README.md) for detailed instructions.
* Extract `Fake.csv` and `True.csv` into the `dataset/` folder.

## 🧠 Usage

### Model Training
Before running the web app, you must train the model and generate the `.pkl` files.
```bash
python src/train_model.py
```
*This will process the data, vectorize the text, train the Logistic Regression model, and output `fake_news_model.pkl` and `tfidf_vectorizer.pkl` into the `model/` directory.*

### Running the Web Web
Launch the Streamlit interactive dashboard:
```bash
streamlit run app.py
```
The application will open automatically in your default browser at `http://localhost:8501`.

### Data Exploration
To view the EDA, class distributions, and word clouds, launch Jupyter Notebook:
```bash
jupyter notebook notebooks/fake_news_eda.ipynb
```

## ⚠️ Known Limitations & Future Work
**AI-Generated Text Classification:** 
Because this model uses TF-IDF and Logistic Regression, it learns to classify based on *stylistic word patterns* (e.g., sensationalism, excessive punctuation, specific capitalized words often found in human-written fake news).
If you generate a perfectly grammatical "Fake News" article using an LLM (like **ChatGPT**), the model will likely classify it as **True News**. This is because AI generators use highly professional, neutral "journalistic" structures that the model associates with real news.
* **Future Improvement:** To detect AI-generated misinformation, the training dataset needs to be expanded to include AI-written text, and the architecture should be upgraded to a context-aware Transformer model (like BERT or RoBERTa).

## 🛠️ Built With
* [Python 3](https://www.python.org/) - Programming Language
* [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) - Data Manipulation
* [NLTK (Natural Language Toolkit)](https://www.nltk.org/) - Text Preprocessing
* [Scikit-Learn](https://scikit-learn.org/stable/) - Machine Learning algorithms and TF-IDF
* [Streamlit](https://streamlit.io/) - Web Dashboard Creation
* [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - Data Visualization

## 👨‍💻 Author
**Your Name/Username**
* Portfolio: [Link Here](#)
* LinkedIn: [Link Here](#)
* GitHub: [Link Here](#)
