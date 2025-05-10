# Phase 1: Traditional NLP Methods for Arabic Text Classification  

# 1. Imports  
import os  
import pandas as pd  
import numpy as np  
import re  
import nltk  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.svm import LinearSVC  
from sklearn.metrics import classification_report, confusion_matrix  
from nltk.corpus import stopwords  
from nltk.stem.isri import ISRIStemmer  
import multiprocessing  

# Download required resources  
try:  
    stopwords.words('arabic')  
except LookupError:  
    nltk.download('stopwords')  

# Define paths for each category  
paths = {  
    'sport': r'C:\Users\heba0\Downloads\articles-sports\articlesSports',  
    'local': r'C:\Users\heba0\Downloads\articles-local\articlesLocal'  
}  

def load_data():  
    texts = []  
    labels = []  

    print("Loading dataset...")  
    for category, folder_path in paths.items():  
        files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]  

        for filename in files:  
            file_path = os.path.join(folder_path, filename)  
            with open(file_path, 'r', encoding='utf-8') as f:  
                text = f.read()  
                texts.append(text)  
                labels.append(category)  

    # Create DataFrame  
    df = pd.DataFrame({'text': texts, 'label': labels})  
    print(f"Dataset Loaded: {len(df)} samples")  
    print(df['label'].value_counts())  
    return df  

# Define preprocessing functions  
def normalize_arabic(text):  
    text = re.sub(r'[إأآا]', 'ا', text)  
    text = re.sub(r'ى', 'ي', text)  
    text = re.sub(r'ؤ', 'و', text)  
    text = re.sub(r'ئ', 'ي', text)  
    text = re.sub(r'ة', 'ه', text)  
    text = re.sub(r'ـ', '', text)  
    return text  

def remove_diacritics(text):  
    arabic_diacritics = re.compile(r"ّ|َ|ً|ُ|ٌ|ِ|ٍ|ْ|ـ")  
    return re.sub(arabic_diacritics, '', text)  

def clean_text(text):  
    text = normalize_arabic(text)  
    text = remove_diacritics(text)  
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation  
    text = re.sub(r'\d+', '', text)      # Remove digits  
    return text  

def preprocess_text(text, stopwords_set, stemmer):  
    text = clean_text(text)  
    words = text.split()  
    words = [word for word in words if word not in stopwords_set]  
    words = [stemmer.stem(word) for word in words]  
    return ' '.join(words)  

def preprocess_data(df):  
    print("Preprocessing text...")  
    
    arabic_stopwords = set(stopwords.words('arabic'))  
    stemmer = ISRIStemmer()  

    with multiprocessing.Pool() as pool:  
        # Pass the stopwords_set and stemmer to the pooled function  
        df['clean_text'] = pool.starmap(preprocess_text, [(text, arabic_stopwords, stemmer) for text in df['text']])  

# 4. Feature Extraction  
# -----------------------------------------------  

def feature_extraction(df):  
    X = df['clean_text']  
    y = df['label']  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  

    # Feature extraction for both BoW and TF-IDF  
    bow_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)  
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)  

    print("Extracting features...")  
    X_train_bow = bow_vectorizer.fit_transform(X_train)  
    X_test_bow = bow_vectorizer.transform(X_test)  

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  
    X_test_tfidf = tfidf_vectorizer.transform(X_test)  

    return (X_train_bow, X_test_bow, y_train, y_test), (X_train_tfidf, X_test_tfidf, y_train, y_test)  

# 5. Traditional Implementations  
# -----------------------------------------------  

def train_and_evaluate(models, X_train, X_test, y_train, y_test, feature_type):  
    results = {}  
    for name, model in models.items():  
        print(f"\n--- Model: {name} ---\n")  
        model.fit(X_train, y_train)  
        y_pred = model.predict(X_test)  

        # Collect evaluation metrics  
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)  
        results[name] = {  
            "precision": report['weighted avg']['precision'],  
            "recall": report['weighted avg']['recall'],  
            "f1-score": report['weighted avg']['f1-score'],  
            "accuracy": np.mean(y_pred == y_test)  
        }  
        print(classification_report(y_test, y_pred, zero_division=1))  

        # Confusion Matrix  
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))  
        plt.figure(figsize=(6, 5))  
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  
                    xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))  
        plt.title(f'Confusion Matrix for {name} ({feature_type})')  
        plt.ylabel('True label')  
        plt.xlabel('Predicted label')  
        plt.savefig(f'{name}_{feature_type}_confusion_matrix.png')  # Save plot as a file  
        plt.close()  

    return results  

if __name__ == '__main__':  
    df = load_data()  
    preprocess_data(df)  
    
    # Feature extraction  
    (X_train_bow, X_test_bow, y_train_bow, y_test_bow), (X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf) = feature_extraction(df)  

    # Define models to be evaluated  
    models = {  
        'Naive Bayes': MultinomialNB(),  
        'SVM': LinearSVC()  
    }  

    # Evaluate Models using Bag of Words  
    print("Evaluating models with BoW + N-grams...")  
    results_bow = train_and_evaluate(models, X_train_bow, X_test_bow, y_train_bow, y_test_bow, feature_type="BoW")  

    # Evaluate Models using TF-IDF  
    print("Evaluating models with TF-IDF + N-grams...")  
    results_tfidf = train_and_evaluate(models, X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf, feature_type="TF-IDF")  

    # Comparison of results  
    print("\nComparison of Model Performance:")  
    print(f"{'Model':<20} {'Metric':<10} {'BoW':<10} {'TF-IDF':<10}")  
    for model in models.keys():  
        print(f"{model:<20} {'Accuracy':<10} {results_bow[model]['accuracy']:.2f} {results_tfidf[model]['accuracy']:.2f}")  
        print(f"{model:<20} {'F1-Score':<10} {results_bow[model]['f1-score']:.2f} {results_tfidf[model]['f1-score']:.2f}")  
        print(f"{model:<20} {'Precision':<10} {results_bow[model]['precision']:.2f} {results_tfidf[model]['precision']:.2f}")  
        print(f"{model:<20} {'Recall':<10} {results_bow[model]['recall']:.2f} {results_tfidf[model]['recall']:.2f}")  
        print("\n")