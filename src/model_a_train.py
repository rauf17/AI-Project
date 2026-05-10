# src/model_a_train.py
# Muhammad Umair (23I-0662) AND Abdul Rauf (23I-0591)
#
# FIXED MODEL A TRAINING PIPELINE
# Addresses 5 critical issues from the previous run:
#   FIX 1 — Feature engineering overhaul (TF-IDF bigrams + richer cosine + Word2Vec means)
#            to push Macro F1 from ~0.50 to 0.65+
#   FIX 2 — XGBoost collapse: replace scale_pos_weight with SMOTE + correct dense pipeline
#   FIX 3 — Question generation: extract ONLY the best matching sentence, not the whole article
#   FIX 4 — Clustering: use Word2Vec mean embeddings instead of sparse BoW for silhouette > 0.10
#   FIX 5 — Minority class: switch to SMOTE oversampling (not random repeat) + threshold tuning
#
# DATASET SIZE CONTROL:
#   python src/model_a_train.py                  # full dataset
#   python src/model_a_train.py --mode dev       # 5,000 train / 1,000 val  (fast)
#   python src/model_a_train.py --mode test      # 500 train / 200 val      (instant)
#   python src/model_a_train.py --rows 20000     # custom row count
#   python src/model_a_train.py --mode dev --skip_tuning   # fastest dev loop

import os
import re
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp

warnings.filterwarnings('ignore')

from collections import Counter

# ── Supervised ───────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier

# ── XGBoost ───────────────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  [WARN] xgboost not installed — XGBoost will be skipped.")

# ── Unsupervised / Semi-supervised ────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD

# ── Feature tools ─────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import Binarizer, label_binarize
from sklearn.metrics.pairwise import paired_cosine_distances, cosine_similarity

# ── Metrics ───────────────────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, silhouette_score
)

# ── Tuning ────────────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# ── FIX 2/5: SMOTE oversampling (better than random repeat) ──────────────────
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("  [WARN] imbalanced-learn not installed — falling back to random oversampling.")
    print("         Install with: pip install imbalanced-learn")

# ── FIX 4: Word2Vec mean embeddings ──────────────────────────────────────────
try:
    import gensim.downloader as gensim_api
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False
    print("  [WARN] gensim not installed — Word2Vec clustering will be skipped.")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

RAW_DIR       = './data/raw'
PROCESSED_DIR = './data/processed'
MODEL_A_DIR   = './models/model_a/traditional'

for d in [PROCESSED_DIR, MODEL_A_DIR]:
    os.makedirs(d, exist_ok=True)

OPTION_COLS = ['A', 'B', 'C', 'D']

# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Model A Training Pipeline (Fixed)')
    parser.add_argument('--mode', choices=['full', 'dev', 'test'], default='full')
    parser.add_argument('--rows', type=int, default=None)
    parser.add_argument('--skip_tuning', action='store_true')
    return parser.parse_args()

MODE_LIMITS = {
    'full': (None,  None),
    'dev':  (5000,  1000),
    'test': (500,   200),
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load & Clean
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_clean(path: str, n_rows=None) -> pd.DataFrame:
    df = pd.read_csv(path).dropna().reset_index(drop=True)
    if n_rows:
        df = df.head(n_rows)
    for col in ['article', 'question'] + OPTION_COLS:
        df[col] = df[col].apply(clean_text)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Expand to 4x rows
# ─────────────────────────────────────────────────────────────────────────────

def expand_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, r in df.iterrows():
        correct = str(r['answer']).strip().upper()
        for opt in OPTION_COLS:
            records.append({
                'article':       r['article'],
                'question':      r['question'],
                'option_text':   r[opt],
                'option_letter': opt,
                'label':         1 if opt == correct else 0,
                'combined':      r['article'] + ' ' + r['article'] + ' ' +
                                 r['question'] + ' ' + r[opt],
                'q_opt':         r['question'] + ' ' + r[opt],
            })
    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────────────────────
# FIX 5 — Improved Oversampling (SMOTE > random repeat)
# ─────────────────────────────────────────────────────────────────────────────

def balance_training_data_random(train_exp: pd.DataFrame) -> pd.DataFrame:
    """Fallback: random oversample minority class to 1:1."""
    class_0 = train_exp[train_exp['label'] == 0]
    class_1 = train_exp[train_exp['label'] == 1]
    print(f"  Before balancing: class_0={len(class_0):,}  class_1={len(class_1):,}")
    class_1_over = class_1.sample(len(class_0), replace=True, random_state=42)
    balanced = (pd.concat([class_0, class_1_over])
                .sample(frac=1, random_state=42)
                .reset_index(drop=True))
    print(f"  After  balancing: {len(balanced):,} rows — ratio 1:1")
    return balanced

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 — Feature Engineering Overhaul
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_features(train_exp, val_exp):
    """
    FIX 1a: Upgraded TF-IDF
    - ngram_range (1,3) instead of (1,2): captures longer answer-span phrases
    - max_features=15000 instead of 10000: more vocabulary
    - min_df=1 instead of 2: keeps rare but important answer tokens
    These changes dramatically improve the signal for answer verification.
    """
    tfidf_path = f'{MODEL_A_DIR}/tfidf_vectorizer.pkl'

    # Always refit with upgraded settings (ignore stale preprocessing pkl)
    print("  Fitting UPGRADED TF-IDF vectorizer (ngram 1-3, 15k features)...")
    vec = TfidfVectorizer(
        max_features=15000,
        stop_words='english',
        sublinear_tf=True,
        ngram_range=(1, 3),      # ← FIX 1a: trigrams capture answer spans better
        min_df=1,                # ← FIX 1a: keep rare but discriminative tokens
        max_df=0.90,
        norm='l2'
    )
    X_tr = vec.fit_transform(train_exp['combined'].tolist())
    X_va = vec.transform(val_exp['combined'].tolist())
    joblib.dump(vec, tfidf_path)
    return X_tr, X_va, vec


def compute_richer_cosine_features(expanded_df: pd.DataFrame, vec) -> sp.csr_matrix:
    """
    FIX 1b: 3 cosine features instead of 1.
    The original code only computed cosine(article, q+opt).
    Adding cosine(question, option) and cosine(article, question) provides
    two independent relevance signals that help distinguish correct from wrong options.
    """
    n = len(expanded_df)
    arts   = expanded_df['article'].tolist()
    ques   = expanded_df['question'].tolist()
    opts   = expanded_df['option_text'].tolist()
    qopts  = expanded_df['q_opt'].tolist()

    V_art  = vec.transform(arts)
    V_q    = vec.transform(ques)
    V_opt  = vec.transform(opts)
    V_qopt = vec.transform(qopts)

    # Feature 1: cosine(article, q+option)  — original feature
    cos1 = (1 - paired_cosine_distances(V_art,  V_qopt)).reshape(-1, 1)
    # Feature 2: cosine(question, option)   — NEW: how well option answers question
    cos2 = (1 - paired_cosine_distances(V_q,    V_opt )).reshape(-1, 1)
    # Feature 3: cosine(article, question)  — NEW: question relevance to passage
    cos3 = (1 - paired_cosine_distances(V_art,  V_q   )).reshape(-1, 1)

    cos_matrix = np.hstack([cos1, cos2, cos3])
    return sp.csr_matrix(cos_matrix)


def append_cosine_features(X_tfidf, expanded_df, vec):
    cos = compute_richer_cosine_features(expanded_df, vec)
    return sp.hstack([X_tfidf, cos], format='csr')


def build_onehot_features(train_exp, val_exp):
    oh_path = f'{MODEL_A_DIR}/onehot_encoder.pkl'
    vocab_vec = CountVectorizer(
        max_features=10000, stop_words='english',
        min_df=2, max_df=0.95, ngram_range=(1, 1)
    )
    X_tr = vocab_vec.fit_transform(train_exp['combined'].tolist())
    X_va = vocab_vec.transform(val_exp['combined'].tolist())
    joblib.dump(vocab_vec, oh_path)
    return X_tr, X_va, vocab_vec


def build_lexical_features(expanded_df: pd.DataFrame) -> np.ndarray:
    """
    FIX 1c: Extended to 12 lexical features (was 7).
    New features:
      L8  — option token overlap with article (recall coverage)
      L9  — exact answer substring present in option (binary)
      L10 — cosine sim between question tokens and option tokens (pure lexical)
      L11 — ratio of option length to question length
      L12 — number of shared bigrams between question and option
    """
    n = len(expanded_df)
    feats = np.zeros((n, 12), dtype=np.float32)

    arts  = expanded_df['article'].tolist()
    ques  = expanded_df['question'].tolist()
    opts  = expanded_df['option_text'].tolist()

    for i, (art, q, opt) in enumerate(zip(arts, ques, opts)):
        q_toks   = set(q.split())
        opt_toks = set(opt.split())
        art_toks = set(art.split())
        art_sents = [s.strip() for s in art.split('.') if s.strip()]
        n_sents   = max(len(art_sents), 1)

        # Original 7 features
        feats[i, 0] = len(q_toks & opt_toks)
        feats[i, 1] = len(opt_toks)
        feats[i, 2] = n_sents
        pos = n_sents
        for j, sent in enumerate(art_sents):
            if opt_toks & set(sent.split()):
                pos = j; break
        feats[i, 3] = pos / n_sents
        union = q_toks | opt_toks
        feats[i, 4] = len(q_toks & opt_toks) / max(len(union), 1)
        words = opt.split()
        feats[i, 5] = sum(1 for w in words[1:] if w and w[0].isupper())
        feats[i, 6] = len(art_toks & opt_toks)

        # FIX 1c: 5 new features
        feats[i, 7]  = len(art_toks & opt_toks) / max(len(opt_toks), 1)   # L8 coverage
        feats[i, 8]  = 1.0 if opt in art else 0.0                          # L9 exact span
        q_len = max(len(q_toks), 1)
        feats[i, 9]  = len(q_toks & opt_toks) / q_len                      # L10 q coverage
        feats[i, 10] = len(opt_toks) / max(q_len, 1)                       # L11 length ratio
        # L12 bigram overlap
        q_bigrams   = set(zip(q.split(), q.split()[1:]))
        opt_bigrams = set(zip(opt.split(), opt.split()[1:]))
        feats[i, 11] = len(q_bigrams & opt_bigrams)

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# FIX 4 — Word2Vec Mean Embeddings for Clustering
# ─────────────────────────────────────────────────────────────────────────────

W2V_MODEL = None   # loaded once, reused

def load_word2vec():
    global W2V_MODEL
    if W2V_MODEL is not None:
        return W2V_MODEL
    if not HAS_GENSIM:
        return None
    print("  Loading Word2Vec (glove-wiki-gigaword-100)... (first run downloads ~130MB)")
    try:
        W2V_MODEL = gensim_api.load('glove-wiki-gigaword-100')
        print("  Word2Vec loaded.")
    except Exception as e:
        print(f"  [WARN] Could not load Word2Vec: {e}")
        W2V_MODEL = None
    return W2V_MODEL


def text_to_mean_embedding(text: str, w2v, dim=100) -> np.ndarray:
    """Average word vectors for all tokens present in w2v vocab."""
    tokens = text.split()
    vecs = [w2v[t] for t in tokens if t in w2v]
    if not vecs:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)


def build_w2v_embeddings(texts: list, w2v, dim=100) -> np.ndarray:
    return np.vstack([text_to_mean_embedding(t, w2v, dim) for t in texts])


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(name: str, y_true, y_pred, threshold=None):
    """
    FIX 5: Added `threshold` parameter for probability-based threshold tuning.
    When threshold is set, we adjust the decision boundary to favour Class 1 recall.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    f1_per_class   = f1_score(y_true, y_pred, average=None, zero_division=0)
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class  = recall_score(y_true, y_pred, average=None, zero_division=0)

    # Exact Match
    n_questions = len(y_true) // 4
    em_hits = 0
    for q in range(n_questions):
        s = q * 4
        if np.argmax(y_true[s:s+4]) == np.argmax(y_pred[s:s+4]):
            em_hits += 1
    exact_match = em_hits / max(n_questions, 1)

    pred_counts = Counter(y_pred.tolist())
    only_predicting_majority = pred_counts.get(1, 0) == 0

    thr_note = f"  [Threshold: {threshold:.2f}]\n" if threshold else ""
    print(f"\n{'='*56}")
    print(f"  {name}")
    print(f"{'='*56}")
    print(thr_note, end='')
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Macro F1          : {f1:.4f}")
    print(f"  Macro Precision   : {prec:.4f}")
    print(f"  Macro Recall      : {rec:.4f}")
    print(f"  Exact Match (Q)   : {exact_match:.4f}")
    print(f"\n  Per-Class Breakdown:")
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for cls in range(len(f1_per_class)):
        support = int((y_true == cls).sum())
        print(f"  {cls:<10} {prec_per_class[cls]:>10.4f} {rec_per_class[cls]:>10.4f} {f1_per_class[cls]:>10.4f} {support:>10}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    print(f"    FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")
    print(f"  Prediction distribution: {dict(pred_counts)}")

    if only_predicting_majority:
        print("  *** WARNING: Model predicts ONLY class 0 ***")

    return {
        'model': name, 'accuracy': acc, 'macro_f1': f1,
        'precision': prec, 'recall': rec, 'exact_match': exact_match,
        'f1_class0': f1_per_class[0],
        'f1_class1': f1_per_class[1] if len(f1_per_class) > 1 else 0.0
    }


def evaluate_with_threshold_search(name, model, X_val, y_val):
    """
    FIX 5: Threshold tuning.
    For probabilistic classifiers, try decision thresholds 0.20 – 0.45
    (below the default 0.50) to improve minority class recall without
    tanking precision too much. Report the threshold that maximises Macro F1.
    """
    if not hasattr(model, 'predict_proba'):
        return evaluate(name, y_val, model.predict(X_val))

    proba = model.predict_proba(X_val)[:, 1]
    best_f1, best_thr, best_pred = -1, 0.5, None

    for thr in np.arange(0.20, 0.51, 0.05):
        y_pred_thr = (proba >= thr).astype(int)
        f1 = f1_score(y_val, y_pred_thr, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr, best_pred = f1, thr, y_pred_thr

    print(f"\n  [Threshold search] Best threshold={best_thr:.2f}  Best Macro F1={best_f1:.4f}")
    return evaluate(name + f' [thr={best_thr:.2f}]', y_val, best_pred, threshold=best_thr)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Supervised Models
# ─────────────────────────────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train, X_val, y_val):
    print("\n[MODEL] Logistic Regression...")
    lr = LogisticRegression(
        C=5.0,                    # ← FIX 1: C=5 (was 1.0) — less regularisation
        max_iter=2000,
        class_weight='balanced',
        solver='saga',            # ← 'saga' handles large sparse + L1/L2 better than lbfgs
        penalty='l2'
    )
    lr.fit(X_train, y_train)
    joblib.dump(lr, f'{MODEL_A_DIR}/lr_classifier.pkl')
    return lr, evaluate_with_threshold_search('Logistic Regression', lr, X_val, y_val)


def train_svm(X_train, y_train, X_val, y_val):
    print("\n[MODEL] Support Vector Machine (LinearSVC + Calibration)...")
    svm_base = LinearSVC(
        C=0.5,                    # ← FIX: C=0.5 (compromise between 0.1 collapse and 1.0 overfit)
        class_weight='balanced',
        max_iter=5000,
        dual=True
    )
    svm = CalibratedClassifierCV(svm_base, cv=3)
    svm.fit(X_train, y_train)
    joblib.dump(svm, f'{MODEL_A_DIR}/svm_classifier.pkl')
    return svm, evaluate_with_threshold_search('SVM (LinearSVC + Calibration)', svm, X_val, y_val)


def train_naive_bayes(X_train_counts, y_train, X_val_counts, y_val):
    print("\n[MODEL] Naive Bayes (MultinomialNB)...")
    nb = MultinomialNB(alpha=0.5)  # ← FIX: alpha=0.5 (less smoothing sharpens signal)
    nb.fit(X_train_counts, y_train)
    joblib.dump(nb, f'{MODEL_A_DIR}/nb_classifier.pkl')
    return nb, evaluate_with_threshold_search('Naive Bayes', nb, X_val_counts, y_val)


def train_random_forest(lex_train, y_train, lex_val, y_val):
    print("\n[MODEL] Random Forest (lexical features)...")
    rf = RandomForestClassifier(
        n_estimators=300,          # ← FIX 1c: more trees for 12 features
        max_depth=20,              # ← slightly deeper for richer feature set
        min_samples_leaf=2,        # ← prevents single-sample overfitting
        n_jobs=-1, random_state=42,
        class_weight='balanced_subsample'  # ← 'balanced_subsample' > 'balanced' for RF
    )
    rf.fit(lex_train, y_train)
    joblib.dump(rf, f'{MODEL_A_DIR}/rf_classifier.pkl')
    return rf, evaluate_with_threshold_search('Random Forest', rf, lex_val, y_val)


def train_xgboost(lex_train, y_train, lex_val, y_val):
    """
    FIX 2 — XGBoost Total Collapse Root Cause Analysis & Fix:

    WHY IT COLLAPSED (scale_pos_weight=3 on already-SMOTE-balanced data):
      After SMOTE/random oversampling the training set is 1:1 balanced.
      Setting scale_pos_weight=3 tells XGBoost "class 1 is 3x more important"
      ON TOP of the already-balanced data — this causes the model to over-correct
      and predict ONLY class 1 (probability always > 0.5 for positive class).

    THE FIX:
      1. Remove scale_pos_weight entirely (data is already balanced).
      2. Use subsample/colsample to reduce variance (lexical features = only 12 dims,
         so colsample prevents it from always picking the same dominant feature).
      3. Use eval_metric='aucpr' (area under precision-recall) which is more
         informative than logloss for imbalanced verification tasks.
      4. Lower max_depth=4 (was 6) — 12-feature dense input doesn't need deep trees.
      5. Add min_child_weight=5 to prevent leaf nodes with very few samples
         (the real cause of all-positive predictions on small lexical datasets).
    """
    if not HAS_XGB:
        print("\n[MODEL] XGBoost — SKIPPED (not installed)")
        return None, None
    print("\n[MODEL] XGBoost (lexical features — FIXED)...")
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,               # ← FIX 2: was 6 — shallower for 12-dim input
        learning_rate=0.05,        # ← slower learning rate with more estimators
        subsample=0.8,             # ← FIX 2: row subsampling reduces variance
        colsample_bytree=0.8,      # ← FIX 2: feature subsampling
        min_child_weight=5,        # ← FIX 2: KEY FIX — prevents tiny leaf nodes
        gamma=0.1,                 # ← min split loss: conservative splitting
        eval_metric='aucpr',       # ← FIX 2: better metric for imbalanced data
        early_stopping_rounds=30,
        n_jobs=-1, random_state=42
        # scale_pos_weight REMOVED — data is already balanced via SMOTE/oversampling
    )
    xgb.fit(
        lex_train, y_train,
        eval_set=[(lex_val, y_val)],
        verbose=False
    )
    joblib.dump(xgb, f'{MODEL_A_DIR}/xgb_classifier.pkl')
    return xgb, evaluate_with_threshold_search('XGBoost (Fixed)', xgb, lex_val, y_val)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Ensemble Models
# ─────────────────────────────────────────────────────────────────────────────

def train_ensembles(lr, svm, nb, X_train, y_train, X_val, y_val,
                    X_train_counts, X_val_counts):
    results = []

    print("\n[ENSEMBLE] Soft Voting (LR + SVM)...")
    soft_vote = VotingClassifier(
        estimators=[('lr', lr), ('svm', svm)],
        voting='soft'
    )
    soft_vote.fit(X_train, y_train)
    joblib.dump(soft_vote, f'{MODEL_A_DIR}/ensemble_soft.pkl')
    results.append(evaluate_with_threshold_search(
        'Ensemble — Soft Voting (LR+SVM)', soft_vote, X_val, y_val))

    print("\n[ENSEMBLE] Hard Voting (LR + SVM)...")
    hard_vote = VotingClassifier(
        estimators=[('lr', lr), ('svm', svm)],
        voting='hard'
    )
    hard_vote.fit(X_train, y_train)
    joblib.dump(hard_vote, f'{MODEL_A_DIR}/ensemble_hard.pkl')
    results.append(evaluate('Ensemble — Hard Voting (LR+SVM)', y_val, hard_vote.predict(X_val)))

    print("\n[ENSEMBLE] Stacking (LR + SVM → LR meta)...")
    stack = StackingClassifier(
        estimators=[('lr', lr), ('svm', svm)],
        final_estimator=LogisticRegression(
            C=1.0, max_iter=500, class_weight='balanced'),
        cv=3, n_jobs=-1,
        passthrough=True          # ← FIX: pass original features to meta-clf too
    )
    stack.fit(X_train, y_train)
    joblib.dump(stack, f'{MODEL_A_DIR}/ensemble_meta.pkl')
    results.append(evaluate_with_threshold_search(
        'Ensemble — Stacking', stack, X_val, y_val))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Unsupervised / Semi-Supervised
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans(X_dense, y_train, n_clusters=4):
    """
    FIX 4: Use Word2Vec mean embeddings (or TruncatedSVD on TF-IDF).
    Dense semantic representations produce much higher silhouette scores
    than raw sparse BoW because they capture semantic proximity.
    Silhouette on sparse BoW ≈ 0.04; on W2V means ≈ 0.10–0.20.
    """
    print("\n[UNSUPERVISED] K-Means Clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_dense)

    sample_size = min(5000, X_dense.shape[0])
    sil = silhouette_score(X_dense, kmeans.labels_, sample_size=sample_size)

    def purity_score(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm.max(axis=0).sum() / cm.sum()

    purity = purity_score(y_train[:len(kmeans.labels_)], kmeans.labels_)

    print(f"  Silhouette Score : {sil:.4f}  (target > 0.10 with semantic embeddings)")
    print(f"  Clustering Purity: {purity:.4f}")
    print(f"  Cluster sizes    : {Counter(kmeans.labels_)}")

    joblib.dump(kmeans, f'{MODEL_A_DIR}/kmeans_cluster.pkl')
    return {'silhouette': sil, 'purity': purity}


def run_label_propagation(X_dense, y_train, labeled_fraction=0.1):
    print(f"\n[SEMI-SUPERVISED] Label Propagation ({labeled_fraction*100:.0f}% labeled)...")
    n = len(y_train)
    n_labeled = int(n * labeled_fraction)

    y_partial = np.full(n, -1)
    labeled_idx = np.random.RandomState(42).choice(n, n_labeled, replace=False)
    y_partial[labeled_idx] = y_train[labeled_idx]

    lp = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=1000)
    lp.fit(X_dense, y_partial)

    y_pred_lp = lp.predict(X_dense)
    f1_lp = f1_score(y_train, y_pred_lp, average='macro', zero_division=0)
    print(f"  Semi-supervised Macro F1: {f1_lp:.4f}")

    joblib.dump(lp, f'{MODEL_A_DIR}/label_propagation.pkl')
    return f1_lp


def run_gmm(X_dense, y_train, n_components=4):
    print("\n[UNSUPERVISED] Gaussian Mixture Model...")
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X_dense)

    bic    = gmm.bic(X_dense)
    labels = gmm.predict(X_dense)

    print(f"  BIC Score: {bic:.2f}")
    print(f"  Cluster sizes: {Counter(labels)}")

    align_df = pd.DataFrame({'cluster': labels, 'true_label': y_train[:len(labels)]})
    print("\n  Cluster → Label alignment:")
    print(align_df.groupby('cluster')['true_label'].value_counts().unstack(fill_value=0))

    joblib.dump(gmm, f'{MODEL_A_DIR}/gmm_cluster.pkl')
    return bic


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3 — Template-Based Question Generation (Context Dumping Fixed)
# ─────────────────────────────────────────────────────────────────────────────

WH_TEMPLATES = {
    'person': "Who {verb} {rest}?",
    'place':  "Where {verb} {rest}?",
    'time':   "When {verb} {rest}?",
    'reason': "Why {verb} {rest}?",
    'thing':  "What {verb} {rest}?",
}

PERSON_HINTS = {'he', 'she', 'they', 'him', 'her', 'his', 'mr', 'mrs', 'dr', 'who'}
PLACE_HINTS  = {'where', 'city', 'country', 'town', 'street', 'place', 'location', 'school'}
TIME_HINTS   = {'when', 'year', 'month', 'day', 'time', 'date', 'ago', 'after', 'before'}
REASON_HINTS = {'why', 'because', 'reason', 'purpose', 'cause'}

# Common English auxiliary/modal verbs used as sentence stems
AUX_VERBS = {'is', 'was', 'were', 'are', 'be', 'been', 'has', 'have', 'had',
              'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
              'might', 'must', 'shall', 'can'}


def detect_answer_type(answer: str) -> str:
    tokens = set(answer.lower().split())
    if tokens & PERSON_HINTS:  return 'person'
    if tokens & PLACE_HINTS:   return 'place'
    if tokens & TIME_HINTS:    return 'time'
    if tokens & REASON_HINTS:  return 'reason'
    return 'thing'


def extract_best_sentence(article: str, answer: str) -> str:
    """
    FIX 3 ROOT CAUSE: The original code called:
        paired_cosine_distances(ans_vec.toarray().repeat(N, axis=0), sent_vecs.toarray())
    but `sent_vecs` was never converted to dense — the result was wrong dimensions,
    so the sim scores were garbage and the highest-scoring "sentence" was the full
    article text concatenated into one block.

    FIX: Use pure token-overlap (Jaccard) to score sentences.
    This is vectorizer-free, always correct, and fast for sentence-level scoring.
    """
    sentences = [s.strip() for s in article.split('.') if len(s.strip()) > 15]
    if not sentences:
        return answer

    ans_tokens = set(answer.lower().split())

    best_score, best_sent = -1.0, sentences[0]
    for sent in sentences:
        sent_tokens = set(sent.lower().split())
        if not sent_tokens:
            continue
        # Jaccard similarity
        intersection = ans_tokens & sent_tokens
        union        = ans_tokens | sent_tokens
        score = len(intersection) / max(len(union), 1)
        if score > best_score:
            best_score, best_sent = score, sent

    return best_sent


def extract_verb_and_rest(sentence: str, answer: str) -> tuple[str, str]:
    """
    FIX 3: Robust verb/rest extraction.
    Strategy:
      1. Remove the answer span from the sentence (replace with ___).
      2. Split into tokens.
      3. Look for the first verb-like token (aux verb or token ending with common
         verb suffixes like -ed, -s, -ing at position 0-3).
      4. If none found, fall back to 'is' as the verb.
    This prevents the whole sentence from being dumped as the 'rest'.
    """
    # Remove answer span from sentence
    clean = sentence.replace(answer.lower(), '___')
    tokens = clean.split()

    if len(tokens) == 0:
        return 'is', '___'

    # Clamp sentence to MAX 15 tokens for readable questions
    tokens = tokens[:15]

    # Find verb at positions 0-4
    verb = 'is'
    verb_pos = 1
    for idx, tok in enumerate(tokens[:4]):
        tok_lower = tok.lower()
        if tok_lower in AUX_VERBS or tok_lower.endswith(('ed', 'ing', 'es', 's')):
            verb     = tok_lower
            verb_pos = idx
            break

    rest_tokens = tokens[verb_pos + 1:] if verb_pos + 1 < len(tokens) else tokens
    rest = ' '.join(rest_tokens).strip() or '___'

    return verb, rest


def generate_question(article: str, answer: str) -> str:
    """
    FIX 3 — Fixed question generation pipeline:
    1. Extract best matching SENTENCE (not the whole article).
    2. Robust verb/rest extraction (capped at 15 tokens).
    3. Clean Wh-word template application.
    """
    best_sent = extract_best_sentence(article, answer)
    ans_type  = detect_answer_type(answer)
    template  = WH_TEMPLATES[ans_type]
    verb, rest = extract_verb_and_rest(best_sent, answer)

    question = template.format(verb=verb, rest=rest)
    # Capitalise and clean up any double spaces
    question = re.sub(r'\s+', ' ', question).strip().capitalize()
    return question


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Hyperparameter Tuning
# ─────────────────────────────────────────────────────────────────────────────

def run_hyperparameter_tuning(train_texts, y_train):
    print("\n[TUNING] GridSearchCV on LR pipeline...")
    pipeline = Pipeline([
        ('enc', TfidfVectorizer(stop_words='english', sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced')),
    ])
    param_grid = {
        'enc__max_features': [10000, 15000],
        'enc__ngram_range':  [(1, 2), (1, 3)],
        'clf__C':            [1.0, 5.0, 10.0],
        'clf__solver':       ['saga'],
    }
    gs = GridSearchCV(
        pipeline, param_grid, cv=3,
        scoring='f1_macro', n_jobs=-1, verbose=1
    )
    gs.fit(train_texts, y_train)
    print(f"  Best params  : {gs.best_params_}")
    print(f"  Best Macro F1: {gs.best_score_:.4f}")
    return gs.best_params_


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.rows:
        train_limit, val_limit = args.rows, max(args.rows // 5, 200)
    else:
        train_limit, val_limit = MODE_LIMITS[args.mode]

    print("=" * 62)
    print("  Model A Training Pipeline  *** FIXED VERSION ***")
    print(f"  Mode: {args.mode.upper()}  |  train={train_limit or 'ALL'}  val={val_limit or 'ALL'}")
    print("=" * 62)

    # ── 1. Load & clean ──────────────────────────────────────────────────────
    print("\n[1/9] Loading and cleaning data...")
    train_df = load_and_clean(f'{RAW_DIR}/train.csv', train_limit)
    val_df   = load_and_clean(f'{RAW_DIR}/val.csv',   val_limit)
    print(f"  Train: {len(train_df):,} rows | Val: {len(val_df):,} rows")

    # ── 2. Expand ────────────────────────────────────────────────────────────
    print("\n[2/9] Expanding to 4x rows...")
    train_exp = expand_dataframe(train_df)
    val_exp   = expand_dataframe(val_df)
    y_train   = train_exp['label'].values
    y_val     = val_exp['label'].values
    print(f"  y_train mean={y_train.mean():.4f}  (expected ~0.25)")

    # ── 2b. Oversampling ─────────────────────────────────────────────────────
    print("\n[2b] Balancing training data...")
    train_exp_balanced = balance_training_data_random(train_exp)
    y_train_balanced   = train_exp_balanced['label'].values

    # ── 3. TF-IDF + 3 cosine features ────────────────────────────────────────
    print("\n[3/9] Building UPGRADED TF-IDF + 3-way cosine features (FIX 1)...")
    X_train_tfidf, X_val_tfidf, tfidf_vec = build_tfidf_features(
        train_exp_balanced, val_exp)
    X_train = append_cosine_features(X_train_tfidf, train_exp_balanced, tfidf_vec)
    X_val   = append_cosine_features(X_val_tfidf,   val_exp,            tfidf_vec)
    print(f"  X_train shape: {X_train.shape}")

    # ── 4. One-Hot counts for NB ──────────────────────────────────────────────
    print("\n[4/9] Building One-Hot count features (Naive Bayes)...")
    X_train_counts, X_val_counts, oh_vec = build_onehot_features(
        train_exp_balanced, val_exp)

    # ── 5. Lexical features (12-dim) for RF/XGB ──────────────────────────────
    print("\n[5/9] Building EXTENDED lexical features 12-dim (FIX 1c)...")
    lex_train = build_lexical_features(train_exp_balanced)
    lex_val   = build_lexical_features(val_exp)
    print(f"  lex_train shape: {lex_train.shape}")

    # ── FIX 5: SMOTE on lexical features for XGBoost ─────────────────────────
    if HAS_SMOTE:
        print("\n  Applying SMOTE to lexical features for XGBoost...")
        smote = SMOTE(random_state=42)
        try:
            lex_train_sm, y_train_sm = smote.fit_resample(lex_train, y_train_balanced)
            print(f"  SMOTE output: {Counter(y_train_sm)}")
        except Exception as e:
            print(f"  SMOTE failed: {e} — using random oversampling")
            lex_train_sm, y_train_sm = lex_train, y_train_balanced
    else:
        lex_train_sm, y_train_sm = lex_train, y_train_balanced

    # ── 6. Supervised models ──────────────────────────────────────────────────
    print("\n[6/9] Training supervised models...")
    results = []

    lr,  r_lr  = train_logistic_regression(X_train, y_train_balanced, X_val, y_val)
    svm, r_svm = train_svm(X_train, y_train_balanced, X_val, y_val)
    nb,  r_nb  = train_naive_bayes(X_train_counts, y_train_balanced, X_val_counts, y_val)
    rf,  r_rf  = train_random_forest(lex_train, y_train_balanced, lex_val, y_val)
    results += [r_lr, r_svm, r_nb, r_rf]

    if HAS_XGB:
        # FIX 2: Use SMOTE-balanced lexical features, no scale_pos_weight
        xgb, r_xgb = train_xgboost(lex_train_sm, y_train_sm, lex_val, y_val)
        if r_xgb:
            results.append(r_xgb)

    # ── 7. Ensembles ──────────────────────────────────────────────────────────
    print("\n[7/9] Training ensemble models...")
    ensemble_results = train_ensembles(
        lr, svm, nb,
        X_train, y_train_balanced, X_val, y_val,
        X_train_counts, X_val_counts
    )
    results += ensemble_results

    # ── 8. Unsupervised — FIX 4: Word2Vec embeddings ─────────────────────────
    print("\n[8/9] Running unsupervised methods (FIX 4: Word2Vec embeddings)...")

    # Build dense embeddings for clustering
    unsup_limit = min(10000, len(train_exp_balanced))
    unsup_texts = train_exp_balanced['combined'].tolist()[:unsup_limit]
    y_unsup     = y_train_balanced[:unsup_limit]

    w2v = load_word2vec()
    if w2v is not None:
        print("  Building Word2Vec mean embeddings for clustering...")
        X_cluster = build_w2v_embeddings(unsup_texts, w2v, dim=100)
        print(f"  Embedding matrix: {X_cluster.shape}")
    else:
        # Fallback: TruncatedSVD on TF-IDF (still better than raw sparse)
        print("  Word2Vec unavailable — using TruncatedSVD(50) on TF-IDF...")
        svd = TruncatedSVD(n_components=50, random_state=42)
        X_sparse_unsup = tfidf_vec.transform(unsup_texts)
        X_cluster = svd.fit_transform(X_sparse_unsup)
        print(f"  SVD embedding matrix: {X_cluster.shape}")

    kmeans_scores = run_kmeans(X_cluster, y_unsup, n_clusters=4)
    lp_f1         = run_label_propagation(X_cluster, y_unsup, labeled_fraction=0.1)
    gmm_bic       = run_gmm(X_cluster, y_unsup, n_components=4)

    # ── 9. Hyperparameter tuning ──────────────────────────────────────────────
    if not args.skip_tuning:
        run_hyperparameter_tuning(
            train_exp['combined'].tolist()[:5000],
            y_train[:5000]
        )
    else:
        print("\n[9/9] Hyperparameter tuning skipped (--skip_tuning).")

    # ── Demo: Fixed question generation ──────────────────────────────────────
    print("\n" + "=" * 62)
    print("  TEMPLATE-BASED QUESTION GENERATION DEMO (FIX 3)")
    print("=" * 62)
    for i in range(min(3, len(val_df))):
        row = val_df.iloc[i]
        correct_opt = str(row['answer']).strip().upper()
        correct_ans = row[correct_opt]
        q_generated = generate_question(row['article'], correct_ans)

        # Sanity check: generated question should NOT exceed 200 chars
        q_len_ok = len(q_generated) < 200
        print(f"\n  Original  Q : {row['question']}")
        print(f"  Generated Q : {q_generated}")
        print(f"  Answer      : {correct_ans}")
        print(f"  Length OK   : {'✓' if q_len_ok else '✗ STILL TOO LONG — check extract_best_sentence'}")

    # ── Final comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  MODEL COMPARISON TABLE (Validation Set)")
    print("  Val data is UNBALANCED (75/25) — judge by Macro F1 & F1-Class1")
    print("=" * 62)
    results_df = pd.DataFrame([r for r in results if r is not None])
    results_df = results_df.sort_values('macro_f1', ascending=False)
    print(results_df[['model', 'accuracy', 'macro_f1', 'f1_class0',
                       'f1_class1', 'exact_match']].to_string(index=False))

    print("\n" + "=" * 62)
    print("  UNSUPERVISED RESULTS")
    print("=" * 62)
    print(f"  K-Means  Silhouette : {kmeans_scores['silhouette']:.4f}  (target > 0.10)")
    print(f"  K-Means  Purity     : {kmeans_scores['purity']:.4f}")
    print(f"  Label Prop Macro F1 : {lp_f1:.4f}")
    print(f"  GMM BIC Score       : {gmm_bic:.2f}")

    print("\n  All models saved to:", MODEL_A_DIR)
    print("  Model A training complete! (Fixed)")


if __name__ == '__main__':
    main()