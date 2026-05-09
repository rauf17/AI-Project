# src/evaluate.py
# Muhammad Umair (23I-0662) AND Abdul Rauf (23I-0591)
#
# Full evaluation script for Model A and Model B.
# Loads saved models + processed features and reports all spec metrics.
#
# Usage:
#   python src/evaluate.py --model_a              # evaluate Model A on val set
#   python src/evaluate.py --model_b              # evaluate Model B on val set
#   python src/evaluate.py --model_a --split test # use test set
#   python src/evaluate.py --model_a --model_b    # evaluate both

import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp

warnings.filterwarnings('ignore')

from collections import Counter
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, r2_score,
    silhouette_score
)
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.decomposition import TruncatedSVD

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR       = './data/raw'
PROCESSED_DIR = './data/processed'
MODEL_A_DIR   = './models/model_a/traditional'
MODEL_B_DIR   = './models/model_b/traditional'

OPTION_COLS = ['A', 'B', 'C', 'D']

# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Model A and/or Model B')
    parser.add_argument('--model_a', action='store_true', help='Evaluate Model A')
    parser.add_argument('--model_b', action='store_true', help='Evaluate Model B')
    parser.add_argument('--split', choices=['val', 'test'], default='val',
                        help='Which split to evaluate on (default: val)')
    parser.add_argument('--rows', type=int, default=None,
                        help='Limit rows for quick evaluation (e.g. --rows 1000)')
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Feature Building (mirrors model_a_train.py)
# ---------------------------------------------------------------------------

def build_features(expanded_df: pd.DataFrame, tfidf_vec, onehot_vec) -> dict:
    """Build all feature sets needed for different models."""
    # TF-IDF + cosine
    X_tfidf = tfidf_vec.transform(expanded_df['combined'].tolist())
    V_art   = tfidf_vec.transform(expanded_df['article'].tolist())
    V_qopt  = tfidf_vec.transform(expanded_df['q_opt'].tolist())
    cos_sim = (1 - paired_cosine_distances(V_art, V_qopt)).reshape(-1, 1)
    X_main  = sp.hstack([X_tfidf, sp.csr_matrix(cos_sim)], format='csr')

    # Count features for Naive Bayes
    X_counts = onehot_vec.transform(expanded_df['combined'].tolist())

    # Lexical features for RF / XGBoost
    n = len(expanded_df)
    lex = np.zeros((n, 7), dtype=np.float32)
    for i, (art, q, opt) in enumerate(zip(
        expanded_df['article'], expanded_df['question'], expanded_df['option_text']
    )):
        q_toks   = set(q.split())
        opt_toks = set(opt.split())
        art_toks = set(art.split())
        art_sents = [s.strip() for s in art.split('.') if s.strip()]
        n_s = max(len(art_sents), 1)

        lex[i, 0] = len(q_toks & opt_toks)
        lex[i, 1] = len(opt_toks)
        lex[i, 2] = n_s
        pos = n_s
        for j, sent in enumerate(art_sents):
            if opt_toks & set(sent.split()):
                pos = j; break
        lex[i, 3] = pos / n_s
        union = q_toks | opt_toks
        lex[i, 4] = len(q_toks & opt_toks) / max(len(union), 1)
        words = opt.split()
        lex[i, 5] = sum(1 for w in words[1:] if w and w[0].isupper())
        lex[i, 6] = len(art_toks & opt_toks)

    return {'main': X_main, 'counts': X_counts, 'lexical': lex}

# ---------------------------------------------------------------------------
# Core Evaluation Function
# ---------------------------------------------------------------------------

def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute and print all Model A metrics from spec Section 6.5 and 10.1:
    Accuracy, Macro F1, Precision, Recall, Exact Match, Confusion Matrix,
    per-class breakdown (teacher requirement).
    """
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    # Per-class metrics (teacher specifically asked for this)
    f1_cls   = f1_score(y_true, y_pred, average=None, zero_division=0)
    prec_cls = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_cls  = recall_score(y_true, y_pred, average=None, zero_division=0)

    # Exact Match — within each group of 4 options, does the predicted
    # correct option match the gold correct option?
    n_q = len(y_true) // 4
    em  = sum(
        np.argmax(y_true[q*4:q*4+4]) == np.argmax(y_pred[q*4:q*4+4])
        for q in range(n_q)
    ) / max(n_q, 1)

    pred_dist = Counter(y_pred.tolist())
    collapsed = pred_dist.get(1, 0) == 0 or pred_dist.get(0, 0) == 0

    print(f"\n{'='*56}")
    print(f"  {name}")
    print(f"{'='*56}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Macro F1        : {f1:.4f}   ← primary metric")
    print(f"  Macro Precision : {prec:.4f}")
    print(f"  Macro Recall    : {rec:.4f}")
    print(f"  Exact Match     : {em:.4f}   ← fraction of questions answered correctly")
    print(f"\n  Per-Class Breakdown:")
    print(f"  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for c in range(len(f1_cls)):
        sup = int((y_true == c).sum())
        print(f"  {c:<8} {prec_cls[c]:>10.4f} {rec_cls[c]:>10.4f} {f1_cls[c]:>10.4f} {sup:>10}")

    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"              Pred 0   Pred 1")
    print(f"  Actual 0  {cm[0,0]:>8}  {cm[0,1]:>7}   (TN / FP)")
    print(f"  Actual 1  {cm[1,0]:>8}  {cm[1,1]:>7}   (FN / TP)")
    print(f"\n  Prediction distribution: {dict(pred_dist)}")

    if collapsed:
        print(f"\n  *** WARNING: Model collapsed — only predicting one class! ***")
        print(f"  *** This means it learned nothing about the minority class. ***")

    # Health check — tell the user clearly if the model is working
    print(f"\n  Health Check:")
    if f1_cls[1] < 0.10:
        print(f"  [FAIL] F1 for class 1 = {f1_cls[1]:.4f}  → model is NOT learning correct answers")
    elif f1 < 0.40:
        print(f"  [WEAK] Macro F1 = {f1:.4f}  → model is barely above random (0.50 is random baseline)")
    elif f1 < 0.55:
        print(f"  [OK]   Macro F1 = {f1:.4f}  → model is learning but not great")
    else:
        print(f"  [GOOD] Macro F1 = {f1:.4f}  → model is working well")

    if em > 0.30:
        print(f"  [GOOD] Exact Match = {em:.4f}  → correctly answering >{em*100:.0f}% of questions")
    elif em > 0.25:
        print(f"  [OK]   Exact Match = {em:.4f}  → near random baseline (0.25)")
    else:
        print(f"  [WEAK] Exact Match = {em:.4f}  → below random baseline (0.25)")

    return {
        'model': name, 'accuracy': acc, 'macro_f1': f1,
        'precision': prec, 'recall': rec, 'exact_match': em,
        'f1_class0': f1_cls[0], 'f1_class1': f1_cls[1] if len(f1_cls) > 1 else 0.0,
        'tp': int(cm[1,1]), 'fp': int(cm[0,1]),
        'tn': int(cm[0,0]), 'fn': int(cm[1,0])
    }

# ---------------------------------------------------------------------------
# Model A Evaluation
# ---------------------------------------------------------------------------

def evaluate_model_a(split: str, rows=None):
    print("\n" + "="*60)
    print(f"  MODEL A EVALUATION  —  split={split.upper()}")
    print("="*60)

    # Load data
    csv_path = f'{RAW_DIR}/val.csv' if split == 'val' else f'{RAW_DIR}/test.csv'
    df = load_and_clean(csv_path, rows)
    exp = expand_dataframe(df)
    y_true = exp['label'].values
    print(f"  Loaded {len(df):,} rows → {len(exp):,} option rows  (y.mean={y_true.mean():.4f})")

    # Load vectorizers
    tfidf_path  = f'{MODEL_A_DIR}/tfidf_vectorizer.pkl'
    onehot_path = f'{MODEL_A_DIR}/onehot_encoder.pkl'

    if not os.path.exists(tfidf_path):
        print(f"\n  [ERROR] {tfidf_path} not found.")
        print("  Run preprocessing first:  python src/preprocessing.py")
        return

    tfidf_vec  = joblib.load(tfidf_path)
    onehot_vec = joblib.load(onehot_path)

    print("  Building features...")
    feats = build_features(exp, tfidf_vec, onehot_vec)

    # Models to evaluate: (name, pkl_filename, feature_key)
    model_configs = [
        ('Logistic Regression',        'lr_classifier.pkl',       'main'),
        ('SVM (LinearSVC)',             'svm_classifier.pkl',      'main'),
        ('Naive Bayes',                 'nb_classifier.pkl',       'counts'),
        ('Random Forest',               'rf_classifier.pkl',       'lexical'),
        ('XGBoost',                     'xgb_classifier.pkl',      'lexical'),
        ('Ensemble — Soft Voting',      'ensemble_soft.pkl',       'main'),
        ('Ensemble — Hard Voting',      'ensemble_hard.pkl',       'main'),
        ('Ensemble — Stacking',         'ensemble_meta.pkl',       'main'),
    ]

    all_results = []
    for name, pkl, feat_key in model_configs:
        pkl_path = f'{MODEL_A_DIR}/{pkl}'
        if not os.path.exists(pkl_path):
            print(f"\n  [SKIP] {name} — {pkl} not found")
            continue
        model = joblib.load(pkl_path)
        X = feats[feat_key]
        y_pred = model.predict(X)
        result = evaluate_model(name, y_true, y_pred)
        all_results.append(result)

    # ---------------------------------------------------------------------------
    # Unsupervised evaluation (spec Section 10.3)
    # ---------------------------------------------------------------------------
    print("\n" + "="*56)
    print("  UNSUPERVISED EVALUATION (spec Section 10.3)")
    print("="*56)

    kmeans_path = f'{MODEL_A_DIR}/kmeans_cluster.pkl'
    svd_path    = f'{MODEL_A_DIR}/kmeans_svd.pkl'

    if os.path.exists(kmeans_path) and os.path.exists(svd_path):
        kmeans = joblib.load(kmeans_path)
        svd    = joblib.load(svd_path)

        limit = min(5000, feats['main'].shape[0])
        X_red = svd.transform(feats['main'][:limit])
        labels = kmeans.predict(X_red)
        y_sub  = y_true[:limit]

        sil = silhouette_score(X_red, labels, sample_size=min(2000, limit))

        # Clustering purity
        cm_clust = confusion_matrix(y_sub, labels % 2)  # map to binary
        purity = cm_clust.max(axis=0).sum() / cm_clust.sum()

        print(f"  K-Means Silhouette Score : {sil:.4f}  (higher=better, >0.3 is decent)")
        print(f"  K-Means Purity           : {purity:.4f}  (higher=better, 1.0=perfect)")
        print(f"  Cluster distribution     : {Counter(labels.tolist())}")
    else:
        print("  K-Means model not found — run model_a_train.py first")

    gmm_path = f'{MODEL_A_DIR}/gmm_cluster.pkl'
    if os.path.exists(gmm_path):
        gmm = joblib.load(gmm_path)
        # BIC was already computed during training — just note it
        print(f"  GMM: model found (BIC was reported during training)")

    # ---------------------------------------------------------------------------
    # Final comparison table
    # ---------------------------------------------------------------------------
    if all_results:
        print("\n" + "="*80)
        print("  FINAL MODEL COMPARISON TABLE")
        print("  Sorted by Macro F1 — this is the primary metric")
        print("="*80)
        res_df = pd.DataFrame(all_results).sort_values('macro_f1', ascending=False)
        cols = ['model', 'accuracy', 'macro_f1', 'f1_class0', 'f1_class1', 'exact_match']
        print(res_df[cols].to_string(index=False))

        best = res_df.iloc[0]
        print(f"\n  Best model: {best['model']}")
        print(f"  Macro F1 = {best['macro_f1']:.4f}  |  Exact Match = {best['exact_match']:.4f}")

        # Random baseline comparison
        print(f"\n  Random baseline (4-class): Accuracy=0.25, ExactMatch=0.25")
        print(f"  Random baseline (binary):  Macro F1≈0.50 (always predicts majority)")
        print(f"\n  How to read these results:")
        print(f"  - Macro F1 > 0.55  = model is meaningfully better than random")
        print(f"  - Exact Match > 0.30 = correctly answering >30% of questions")
        print(f"  - F1_class1 > 0.35 = model is actually identifying correct answers")

# ---------------------------------------------------------------------------
# Model B Evaluation (spec Section 7.8 and 10.2)
# ---------------------------------------------------------------------------

def evaluate_model_b(split: str, rows=None):
    print("\n" + "="*60)
    print(f"  MODEL B EVALUATION  —  split={split.upper()}")
    print("="*60)

    csv_path = f'{RAW_DIR}/val.csv' if split == 'val' else f'{RAW_DIR}/test.csv'
    df = load_and_clean(csv_path, rows)
    print(f"  Loaded {len(df):,} rows")

    ranker_path = f'{MODEL_B_DIR}/distractor_ranker.pkl'
    hint_path   = f'{MODEL_B_DIR}/hint_scorer.pkl'

    if not os.path.exists(ranker_path):
        print(f"\n  [SKIP] Model B not trained yet — run model_b_train.py first")
        return

    tfidf_vec = joblib.load(f'{MODEL_A_DIR}/tfidf_vectorizer.pkl')
    ranker    = joblib.load(ranker_path)

    # ---------------------------------------------------------------------------
    # Distractor evaluation: for each row, rank all 4 options and check if the
    # top-3 predicted distractors (non-correct options) are retrieved correctly
    # ---------------------------------------------------------------------------
    print("\n  Evaluating distractor ranker...")

    y_distractor_true = []
    y_distractor_pred = []
    precision_at_3_hits = 0

    sample = df.head(min(500, len(df)))  # sample for speed

    for _, row in sample.iterrows():
        correct = str(row['answer']).strip().upper()

        # Build features for all 4 options
        opt_feats = []
        for opt in OPTION_COLS:
            art_vec = tfidf_vec.transform([row['article']])
            opt_vec = tfidf_vec.transform([row[opt]])
            cos     = float(1 - paired_cosine_distances(art_vec, opt_vec)[0])
            opt_len = len(row[opt].split())
            art_len = max(len(row['article'].split()), 1)
            opt_feats.append([cos, opt_len, opt_len / art_len])

        X_opts = np.array(opt_feats)

        # Label: 0=correct answer (should NOT be a distractor), 1=distractor
        labels_true = [0 if o == correct else 1 for o in OPTION_COLS]

        try:
            labels_pred = ranker.predict(X_opts)
        except Exception:
            continue

        y_distractor_true.extend(labels_true)
        y_distractor_pred.extend(labels_pred.tolist())

        # Precision@3: top-3 predicted distractors should all be non-correct
        pred_distractor_idx = np.argsort(labels_pred)[-3:]
        true_distractors    = {i for i, l in enumerate(labels_true) if l == 1}
        hits = len(set(pred_distractor_idx) & true_distractors)
        precision_at_3_hits += hits / 3

    if y_distractor_true:
        y_dt = np.array(y_distractor_true)
        y_dp = np.array(y_distractor_pred)

        prec = precision_score(y_dt, y_dp, average='macro', zero_division=0)
        rec  = recall_score(y_dt, y_dp, average='macro', zero_division=0)
        f1   = f1_score(y_dt, y_dp, average='macro', zero_division=0)
        acc  = accuracy_score(y_dt, y_dp)
        p3   = precision_at_3_hits / max(len(sample), 1)

        print(f"\n  Distractor Ranker Metrics (spec Section 7.8):")
        print(f"  Precision    : {prec:.4f}")
        print(f"  Recall       : {rec:.4f}")
        print(f"  Macro F1     : {f1:.4f}")
        print(f"  Accuracy     : {acc:.4f}")
        print(f"  Precision@3  : {p3:.4f}  (fraction of top-3 that are valid distractors)")
        print(f"\n  Confusion Matrix:")
        cm = confusion_matrix(y_dt, y_dp)
        print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
        print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    # Hint scorer R² (if available)
    if os.path.exists(hint_path):
        print(f"\n  Hint scorer found — R² would require sentence-level relevance labels")
        print(f"  (generated during model_b_train.py and stored as evaluation artefact)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not args.model_a and not args.model_b:
        print("Specify --model_a and/or --model_b")
        print("Example: python src/evaluate.py --model_a --split val")
        return

    if args.model_a:
        evaluate_model_a(args.split, args.rows)

    if args.model_b:
        evaluate_model_b(args.split, args.rows)

    print("\n  Evaluation complete.")


if __name__ == '__main__':
    main()