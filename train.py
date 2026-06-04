import pandas as pd
import pickle
import os
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download the NLTK stop words (only needs to run once)
nltk.download('stopwords')

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# --- NEW: Text Cleaning Function ---
def clean_text(text):
    # 1. Convert text to lowercase
    text = text.lower()
    
    # 2. Remove punctuation (like ! ? , .)
    text = "".join([char for char in text if char not in string.punctuation])
    
    # 3. Remove stop words ("the", "is", "in")
    stop_words = stopwords.words('english')
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text
# -----------------------------------

print("Loading dataset...")
# Load dataset
df = pd.read_csv("dataset/spam.csv", encoding='latin-1')

# Keeping only necessary columns 
df = df.iloc[:, [0, 1]]
df.columns = ['label', 'text']

print("Cleaning the email text... (This might take a few seconds)")
# Apply our new cleaning function to every email in the dataset
df['text'] = df['text'].apply(clean_text)

# Split data into Features (X) and Target Label (y)
X = df['text']
y = df['label']

# Split into training data (80%) and testing data (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Converting text to numbers (Vectorization)...")
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)

print("Training the Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Save the trained model and vectorizer
with open('models/spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('models/vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Success! Upgraded model trained and saved inside the 'models/' folder.")