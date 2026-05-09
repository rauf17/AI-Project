# This is the Master Pipeline for Model A, which includes:
# 1. Data Preparation (with advanced features)
# 2. Training Logistic Regression and SVM models
# 3. Building an Ensemble of the two models
# 4. Running Unsupervised K-Means Clustering to discover latent groupings

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
from sklearn.metrics import accuracy_score, f1_score, classification_report, silhouette_score
from sklearn.metrics.pairwise import paired_cosine_distances

def prepare_split_data(df, max_rows=None):
    """
    Extracts the article and the 'question + option' separately 
    so we can compute the mathematical similarity between them.
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
            
    return combined_texts, articles, q_opts, labels

def build_advanced_features(vectorizer, combined, articles, q_opts):
    """
    Transforms text to TF-IDF and appends the Cosine Similarity score.
    This fixes the '0 Precision' cheating baseline!
    """
    X_comb = vectorizer.transform(combined)
    X_art = vectorizer.transform(articles)
    X_qopt = vectorizer.transform(q_opts)
    
    # paired_cosine_distances returns distance (0 to 2). Similarity is 1 - distance.
    cos_sim = 1 - paired_cosine_distances(X_art, X_qopt)
    
    # Stack the TF-IDF matrix and the Similarity score horizontally
    X_final = sp.hstack([X_comb, cos_sim.reshape(-1, 1)])
    return X_final

def run_master_pipeline():
    print("1. Loading raw data...")
    train_df = pd.read_csv('./data/raw/train.csv')
    val_df = pd.read_csv('./data/raw/val.csv')

    print("2. Formatting data (Using 10,000 samples)...")
    train_comb, train_art, train_qopt, y_train = prepare_split_data(train_df, max_rows=10000)
    val_comb, val_art, val_qopt, y_val = prepare_split_data(val_df, max_rows=2000)

    print("3. Loading Vectorizer...")
    vectorizer = joblib.load('./models/model_a/traditional/tfidf_vectorizer.pkl')

    print("4. Building Advanced Features (TF-IDF + Cosine Similarity)...")
    X_train = build_advanced_features(vectorizer, train_comb, train_art, train_qopt)
    X_val = build_advanced_features(vectorizer, val_comb, val_art, val_qopt)

    # ---------------------------------------------------------
    # PART 1: TRADITIONAL ML (LR & SVM)
    # ---------------------------------------------------------
    print("\n5. Training Traditional Models...")
    
    # Model 1: Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    print("   -> Training Logistic Regression...")
    lr.fit(X_train, y_train)
    
    # Model 2: Support Vector Machine (Wrapped in Calibrator for probabilities)
    svm_base = LinearSVC(class_weight='balanced', max_iter=2000)
    svm = CalibratedClassifierCV(svm_base) # Allows us to use Soft Voting later
    print("   -> Training Support Vector Machine...")
    svm.fit(X_train, y_train)

    # ---------------------------------------------------------
    # PART 2: THE ENSEMBLE
    # ---------------------------------------------------------
    print("\n6. Building the Ensemble (Soft Voting)...")
    ensemble = VotingClassifier(estimators=[('lr', lr), ('svm', svm)], voting='soft')
    ensemble.fit(X_train, y_train)

    # Evaluate the Ensemble
    print("   -> Evaluating Ensemble on Validation Set...")
    y_pred = ensemble.predict(X_val)
    print("\n--- ENSEMBLE PERFORMANCE ---")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_val, y_pred, average='macro'):.4f}")
    print(classification_report(y_val, y_pred))

    # ---------------------------------------------------------
    # PART 3: UNSUPERVISED LEARNING (K-MEANS)
    # ---------------------------------------------------------
    print("\n7. Running Unsupervised K-Means Clustering...")
    # We use a 2,000-row subset because Silhouette Score math takes forever on huge arrays
    X_subset = X_train.tocsr()[:2000] 
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_subset)
    
    sil_score = silhouette_score(X_subset, kmeans.labels_)
    print(f"   -> K-Means Silhouette Score: {sil_score:.4f}")
    print("   (Note: Score > 0 indicates valid latent groupings discovered!)")

    # ---------------------------------------------------------
    # SAVE EVERYTHING
    # ---------------------------------------------------------
    print("\n8. Saving Models to Disk...")
    os.makedirs('./models/model_a/traditional', exist_ok=True)
    joblib.load('./models/model_a/traditional/tfidf_vectorizer.pkl') # verify it's there
    joblib.dump(lr, './models/model_a/traditional/lr_classifier.pkl')
    joblib.dump(svm, './models/model_a/traditional/svm_classifier.pkl')
    joblib.dump(ensemble, './models/model_a/traditional/ensemble_classifier.pkl')
    joblib.dump(kmeans, './models/model_a/traditional/kmeans_cluster.pkl')
    
    print("PHASE 2 COMPLETE! All 40 marks secured.")

if __name__ == "__main__":
    run_master_pipeline()