import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import scipy.sparse as sp
import os

def load_data():
    print("Loading raw datasets from CSV files...")
    train = pd.read_csv('./data/raw/train.csv').dropna()
    val = pd.read_csv('./data/raw/val.csv').dropna()
    test = pd.read_csv('./data/raw/test.csv').dropna()
    return train, val, test

def build_features():
    train_df, val_df, test_df = load_data()
    
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    print("Fitting vectorizer on training articles...")
    X_train = vectorizer.fit_transform(train_df['article'])
    
    print("Transforming validation and test articles...")
    X_val = vectorizer.transform(val_df['article'])
    X_test = vectorizer.transform(test_df['article'])
    
    # Ensure processed directory exists using the correct relative paths
    os.makedirs('./data/processed', exist_ok=True)
    os.makedirs('./models/model_a/traditional', exist_ok=True)
    
    print("Saving vectorizer and sparse matrices...")
    joblib.dump(vectorizer, './models/model_a/traditional/tfidf_vectorizer.pkl')
    sp.save_npz('./data/processed/X_train_tfidf.npz', X_train)
    sp.save_npz('./data/processed/X_val_tfidf.npz', X_val)
    sp.save_npz('./data/processed/X_test_tfidf.npz', X_test)
    
    print("Preprocessing complete. Ready for Model A.")

if __name__ == "__main__":
    build_features()