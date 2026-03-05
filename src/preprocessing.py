import re
import string
import nltk

# We will download the stopwords within the script but typically 
# it's good practice to do it once. 
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize the stemmer and set of English stop words
stemmer = PorterStemmer()
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    # Fallback in case the download fails in a restricted environment
    stop_words = set()

def clean_text(text):
    """
    Cleans the input text for NLP tasks.
    
    Steps performed:
    1. Lowercase the text.
    2. Remove URLs.
    3. Remove HTML tags.
    4. Remove punctuation.
    5. Remove numbers.
    6. Remove stopwords.
    7. Apply Stemming.
    
    Args:
        text (str): The raw input string.
        
    Returns:
        str: The cleaned string.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase text
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    
    # 5. Remove numbers
    text = re.sub(r'\w*\d\w*', '', text)
    
    # 6. Remove stopwords & 7. Apply Stemming
    words = text.split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(cleaned_words)
