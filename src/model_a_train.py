# This is the Master Pipeline for Model A.
# Muhammad Umair (23I-0662) AND Abdul Rauf (23I-0591)

import pandas as pd
import joblib
import numpy as np
import scipy.sparse as sp
import os

# Machine Learning Imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, classification_report, silhouette_score, confusion_matrix
from sklearn.metrics.pairwise import paired_cosine_distances

def prepare_split_data(df, max_rows=None, balance_data=False):
    """
    Extracts texts and applies Oversampling if balance_data=True to fix class imbalance.
    """
    if max_rows: df = df.head(max_rows)
        
    combined_texts, articles, q_opts, labels = [], [], [], []
    
    for _, row in df.iterrows():
        art = str(row['article'])
        q = str(row['question'])
        ans = str(row['answer'])
        
        for opt_letter in ['A', 'B', 'C', 'D']:
            opt_txt = str(row[opt_letter])
            articles.append(art)
            q_opts.append(f"{q} {opt_txt}")
            combined_texts.append(f"{art} {q} {opt_txt}")
            labels.append(1 if opt_letter == ans else 0)
            
    # --- PROFESSOR'S FIX: OVERSAMPLING ---
    # If this is training data, balance the 0s and 1s perfectly
    if balance_data:
        temp_df = pd.DataFrame({'comb': combined_texts, 'art': articles, 'qopt': q_opts, 'label': labels})
        class_0 = temp_df[temp_df['label'] == 0]
        class_1 = temp_df[temp_df['label'] == 1]
        
        # Duplicate the correct answers so they match the amount of wrong answers
        class_1_over = class_1.sample(len(class_0), replace=True, random_state=42)
        
        # Shuffle them back together
        balanced_df = pd.concat([class_0, class_1_over]).sample(frac=1, random_state=42)
        
        return balanced_df['comb'].tolist(), balanced_df['art'].tolist(), balanced_df['qopt'].tolist(), balanced_df['label'].tolist()

    return combined_texts, articles, q_opts, labels

def build_advanced_features(vectorizer, combined, articles, q_opts):
    """Transforms text to TF-IDF and appends the Cosine Similarity score."""
    X_comb = vectorizer.transform(combined)
    X_art = vectorizer.transform(articles)
    X_qopt = vectorizer.transform(q_opts)
    
    # Cosine Similarity (1 - distance)
    cos_sim = 1 - paired_cosine_distances(X_art, X_qopt)
    
    # Stack TF-IDF matrix and Similarity score horizontally
    X_final = sp.hstack([X_comb, cos_sim.reshape(-1, 1)])
    return X_final

def run_master_pipeline():
    print("1. Loading raw data...")
    train_df = pd.read_csv('./data/raw/train.csv')
    val_df = pd.read_csv('./data/raw/val.csv')

    print("2. Formatting data & Balancing Classes (Oversampling)...")
    # Notice balance_data=True for training only!
    train_comb, train_art, train_qopt, y_train = prepare_split_data(train_df, max_rows=10000, balance_data=True)
    # Validation data stays imbalanced because that's how it is in the real world
    val_comb, val_art, val_qopt, y_val = prepare_split_data(val_df, max_rows=2000, balance_data=False)

    print("3. Loading Vectorizer...")
    vectorizer = joblib.load('./models/model_a/traditional/tfidf_vectorizer.pkl')

    print("4. Building Advanced Features (TF-IDF + Cosine Similarity)...")
    X_train = build_advanced_features(vectorizer, train_comb, train_art, train_qopt)
    X_val = build_advanced_features(vectorizer, val_comb, val_art, val_qopt)

    print("\n5. Training Traditional Models on Balanced Data...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    print("   -> Training Logistic Regression...")
    lr.fit(X_train, y_train)
    
    svm_base = LinearSVC(class_weight='balanced', max_iter=2000)
    svm = CalibratedClassifierCV(svm_base) 
    print("   -> Training Support Vector Machine...")
    svm.fit(X_train, y_train)

    print("\n6. Building the Ensemble (Soft Voting)...")
    ensemble = VotingClassifier(estimators=[('lr', lr), ('svm', svm)], voting='soft')
    ensemble.fit(X_train, y_train)

    print("   -> Evaluating Ensemble on Validation Set...")
    y_pred = ensemble.predict(X_val)
    
    print("\n======================================")
    print("       ENSEMBLE PERFORMANCE")
    print("======================================")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_val, y_pred, average='macro'):.4f}")
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_val, y_pred)
    print(f"True Negatives (Guessed Wrong Option correctly)  : {cm[0][0]}")
    print(f"False Positives (Guessed Wrong Option as right)  : {cm[0][1]}")
    print(f"False Negatives (Guessed Right Option as wrong)  : {cm[1][0]}")
    print(f"True Positives (Guessed Right Option correctly)  : {cm[1][1]}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_val, y_pred))

    print("\n7. Running Unsupervised K-Means Clustering...")
    X_subset = X_train.tocsr()[:2000] 
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_subset)
    print(f"   -> K-Means Silhouette Score: {silhouette_score(X_subset, kmeans.labels_):.4f}")

    print("\n8. Saving Models to Disk...")
    os.makedirs('./models/model_a/traditional', exist_ok=True)
    joblib.dump(lr, './models/model_a/traditional/lr_classifier.pkl')
    joblib.dump(svm, './models/model_a/traditional/svm_classifier.pkl')
    joblib.dump(ensemble, './models/model_a/traditional/ensemble_classifier.pkl')
    joblib.dump(kmeans, './models/model_a/traditional/kmeans_cluster.pkl')
    
    print("PHASE 2 COMPLETE! Models saved and metrics generated for the professor.")

if __name__ == "__main__":
    run_master_pipeline()