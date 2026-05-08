import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os

def prepare_binary_data(df, max_rows=None):
    """
    Converts RACE multi-choice format into binary classification format.
    Every question becomes 4 rows (1 correct option, 3 wrong options).
    """
    if max_rows:
        df = df.head(max_rows)
        
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        article = str(row['article'])
        question = str(row['question'])
        correct_ans = str(row['answer']) # 'A', 'B', 'C', or 'D'
        
        for option_letter in ['A', 'B', 'C', 'D']:
            option_text = str(row[option_letter])
            
            # The manual (Sec 6.1) recommends double-weighting the article
            combined_text = f"{article} {article} {question} {option_text}"
            texts.append(combined_text)
            
            # Label is 1 if this option is the correct answer, else 0
            labels.append(1 if option_letter == correct_ans else 0)
            
    return texts, labels

def train_model_a():
    print("1. Loading raw data from CSV files")
    train_df = pd.read_csv('./data/raw/train.csv')
    val_df = pd.read_csv('./data/raw/val.csv')

    print("2. Formatting data for binary classification (Subsampling for speed) from CSV files")
    # Using 10,000 rows (which becomes 40,000 training samples)
    train_texts, y_train = prepare_binary_data(train_df, max_rows=10000)
    val_texts, y_val = prepare_binary_data(val_df, max_rows=2000)

    print("3. Loading your saved TF-IDF Vectorizer from CSV files")
    vectorizer = joblib.load('./models/model_a/traditional/tfidf_vectorizer.pkl')

    print("4. Vectorizing the text (this takes a moment) from CSV files")
    # IMPORTANT: We use transform() here, not fit_transform(), to prevent data leakage!
    X_train = vectorizer.transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    print("5. Training Logistic Regression Answer Verifier from CSV files")
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)

    print("6. Evaluating on Validation Set from CSV files")
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    
    print(f"\n--- Model A Performance ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    print("7. Saving trained model from CSV files")
    joblib.dump(clf, './models/model_a/traditional/lr_classifier.pkl')
    print("Done! Model A is ready for inference.")

if __name__ == "__main__":
    train_model_a()