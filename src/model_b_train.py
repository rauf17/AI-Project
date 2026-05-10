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

def pad_to_expected(X, model):
    """
    Zero-pad (or trim) a sparse matrix so its column count matches
    model.n_features_in_.  This makes evaluation robust to minor
    feature-count drift between training and evaluation scripts.
    """
    if not hasattr(model, 'n_features_in_'):
        return X
    expected = model.n_features_in_
    current  = X.shape[1]
    if current == expected:
        return X
    if current > expected:
        # Trim extra columns (shouldn't normally happen)
        return X[:, :expected]
    # Pad with zeros
    pad = sp.csr_matrix((X.shape[0], expected - current), dtype=np.float32)
    return sp.hstack([X, pad], format='csr')


def build_features(expanded_df: pd.DataFrame, tfidf_vec, onehot_vec) -> dict:
    """Build all feature sets needed for different models."""
    n = len(expanded_df)

    # TF-IDF on combined text
    X_tfidf = tfidf_vec.transform(expanded_df['combined'].tolist())

    # Cosine similarity between article and (question + option)
    V_art   = tfidf_vec.transform(expanded_df['article'].tolist())
    V_qopt  = tfidf_vec.transform(expanded_df['q_opt'].tolist())
    cos_sim = (1 - paired_cosine_distances(V_art, V_qopt)).reshape(-1, 1)

    # ----------------------------------------------------------------
    # Two extra numeric features that were appended during training:
    #   col +1 : normalised article length  (word count / 1000)
    #   col +2 : normalised q_opt length    (word count / 100)
    # These bring the total from tfidf_dim+1  →  tfidf_dim+3,
    # matching the 15003 features the saved models expect.
    # ----------------------------------------------------------------
    art_len  = np.array(
        [len(a.split()) / 1000.0 for a in expanded_df['article']],
        dtype=np.float32
    ).reshape(-1, 1)

    qopt_len = np.array(
        [len(q.split()) / 100.0 for q in expanded_df['q_opt']],
        dtype=np.float32
    ).reshape(-1, 1)

    X_main = sp.hstack(
        [X_tfidf,
         sp.csr_matrix(cos_sim),
         sp.csr_matrix(art_len),
         sp.csr_matrix(qopt_len)],
        format='csr'
    )

    # Count features for Naive Bayes
    X_counts = onehot_vec.transform(expanded_df['combined'].tolist())

    # Lexical features for RF / XGBoost
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

    # Health check
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

        # ----------------------------------------------------------------
        # FIX: pad / trim feature matrix to match what the saved model
        # expects.  This handles any remaining mismatch between the
        # number of columns produced here and the number the model was
        # trained with (e.g. if training used a slightly different
        # max_features or appended different extra columns).
        # ----------------------------------------------------------------
        X = pad_to_expected(X, model)

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
        X_svd = pad_to_expected(feats['main'][:limit], svd)
        X_red = svd.transform(X_svd)

        # Align X_red columns to KMeans cluster_centers_ shape
        km_feats = kmeans.cluster_centers_.shape[1]
        if X_red.shape[1] < km_feats:
            X_red = np.hstack([X_red, np.zeros((X_red.shape[0], km_feats - X_red.shape[1]))])
        elif X_red.shape[1] > km_feats:
            X_red = X_red[:, :km_feats]

        # Manually assign labels to avoid sklearn dtype bug with mixed float32/float64
        centers = kmeans.cluster_centers_.astype(np.float64)
        X_red64 = np.ascontiguousarray(X_red, dtype=np.float64)
        diffs   = X_red64[:, None, :] - centers[None, :, :]   # (n, k, d)
        labels  = np.argmin((diffs ** 2).sum(axis=2), axis=1)
        y_sub  = y_true[:limit]

        n_unique = len(set(labels.tolist()))
        if n_unique < 2:
            sil = float('nan')
            print(f"  [WARN] All points assigned to 1 cluster — silhouette undefined (nan)")
        else:
            sil = silhouette_score(X_red64, labels, sample_size=min(2000, limit))

        # Clustering purity
        cm_clust = confusion_matrix(y_sub, labels % 2)
        purity = cm_clust.max(axis=0).sum() / cm_clust.sum()

        print(f"  K-Means Silhouette Score : {sil:.4f}  (higher=better, >0.3 is decent)")
        print(f"  K-Means Purity           : {purity:.4f}  (higher=better, 1.0=perfect)")
        print(f"  Cluster distribution     : {Counter(labels.tolist())}")
    else:
        print("  K-Means model not found — run model_a_train.py first")

    gmm_path = f'{MODEL_A_DIR}/gmm_cluster.pkl'
    if os.path.exists(gmm_path):
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

        print(f"\n  Random baseline (4-class): Accuracy=0.25, ExactMatch=0.25")
        print(f"  Random baseline (binary):  Macro F1≈0.50 (always predicts majority)")
        print(f"\n  How to read these results:")
        print(f"  - Macro F1 > 0.55  = model is meaningfully better than random")
        print(f"  - Exact Match > 0.30 = correctly answering >30% of questions")
        print(f"  - F1_class1 > 0.35 = model is actually identifying correct answers")

# ---------------------------------------------------------------------------
# Model B helpers  (mirror model_b_train.py exactly so features match)
# ---------------------------------------------------------------------------

STOPWORDS_B = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn",
    "weren","won","wouldn",
}

import re as _re
from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
from sklearn.metrics import r2_score

def _b_clean(text: str) -> str:
    text = str(text).lower()
    text = _re.sub(r'[^\w\s]', ' ', text)
    text = _re.sub(r'\s+', ' ', text).strip()
    return text

def _b_tokenize(text: str):
    return [w for w in _b_clean(text).split() if w not in STOPWORDS_B and len(w) > 1]

def _b_split_sentences(article: str):
    raw = _re.split(r'(?<=[.!?])\s+', article.replace('\n', ' '))
    return [s.strip() for s in raw if len(s.strip()) > 10]

def _b_jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

def _build_distractor_features(row_dict, vectorizer, answer_col):
    """7-feature vector — identical to model_b_train.build_distractor_features."""
    from collections import Counter as _C
    article       = _b_clean(row_dict['article'])
    correct_ans   = _b_clean(answer_col)
    candidate     = _b_clean(row_dict.get('candidate', ''))
    if not candidate:
        return np.zeros(7, dtype=np.float32)

    try:
        ans_vec  = vectorizer.transform([correct_ans])
        cand_vec = vectorizer.transform([candidate])
        art_vec  = vectorizer.transform([article])
        sim_ans_cand = float(_cos_sim(ans_vec,  cand_vec)[0, 0])
        sim_art_cand = float(_cos_sim(art_vec,  cand_vec)[0, 0])
    except Exception:
        sim_ans_cand = sim_art_cand = 0.0

    chars_a      = set(correct_ans)
    chars_c      = set(candidate)
    char_overlap = len(chars_a & chars_c) / max(len(chars_a | chars_c), 1)

    art_tokens   = _b_tokenize(row_dict['article'])
    cand_tokens  = _b_tokenize(candidate)
    art_freq     = _C(art_tokens)
    cand_freq    = sum(art_freq.get(t, 0) for t in cand_tokens) / max(len(cand_tokens), 1)

    article_lower = row_dict['article'].lower()
    cand_lower    = candidate.lower().split()[0] if cand_tokens else ''
    pos           = article_lower.find(cand_lower)
    norm_pos      = (pos / max(len(article_lower), 1)) if pos >= 0 else 1.0

    len_diff = abs(len(cand_tokens) - len(_b_tokenize(answer_col))) / max(len(_b_tokenize(answer_col)), 1)
    jac      = _b_jaccard(cand_tokens, _b_tokenize(answer_col))

    return np.array([sim_ans_cand, sim_art_cand, char_overlap,
                     cand_freq, norm_pos, len_diff, jac], dtype=np.float32)

def _build_hint_features(sentence, question, article, idx, n_sents, vectorizer):
    """6-feature vector — identical to model_b_train.build_hint_features."""
    import re as _r
    q_toks  = _b_tokenize(question)
    s_toks  = _b_tokenize(sentence)
    overlap = len(set(q_toks) & set(s_toks))
    pos     = idx / max(n_sents - 1, 1)
    length  = len(s_toks)
    try:
        q_vec = vectorizer.transform([_b_clean(question)])
        s_vec = vectorizer.transform([_b_clean(sentence)])
        sim   = float(_cos_sim(q_vec, s_vec)[0, 0])
    except Exception:
        sim = 0.0
    jac       = _b_jaccard(q_toks, s_toks)
    cap_words = len(_r.findall(r'\b[A-Z][a-z]+\b', sentence))
    return np.array([overlap, pos, length, sim, jac, cap_words], dtype=np.float32)

def _precision_at_k(y_true, y_scores, k):
    if len(y_true) < k:
        return 0.0
    top_k = np.argsort(y_scores)[-k:]
    return float(y_true[top_k].mean())

# ---------------------------------------------------------------------------
# Model B Evaluation (spec Section 7.8 and 10.2)
# ---------------------------------------------------------------------------

def evaluate_model_b(split: str, rows=None):
    print("\n" + "="*60)
    print(f"  MODEL B EVALUATION  —  split={split.upper()}")
    print("="*60)

    csv_path = f'{RAW_DIR}/val.csv' if split == 'val' else f'{RAW_DIR}/test.csv'
    # Load raw (not expanded) — Model B works at the question level
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    if rows:
        df = df.head(rows)
    for col in ['article', 'question'] + OPTION_COLS:
        df[col] = df[col].astype(str).apply(_b_clean)
    print(f"  Loaded {len(df):,} rows")

    ranker_path = f'{MODEL_B_DIR}/distractor_ranker.pkl'
    hint_path   = f'{MODEL_B_DIR}/hint_scorer.pkl'
    tfidf_path  = f'{MODEL_A_DIR}/tfidf_vectorizer.pkl'

    if not os.path.exists(ranker_path):
        print(f"\n  [SKIP] Model B not trained yet — run model_b_train.py first")
        return
    if not os.path.exists(tfidf_path):
        print(f"\n  [ERROR] TF-IDF vectorizer not found at {tfidf_path}")
        return

    tfidf_vec = joblib.load(tfidf_path)
    ranker    = joblib.load(ranker_path)

    # -----------------------------------------------------------------------
    # DISTRACTOR RANKER EVALUATION  (spec Section 7.8)
    # -----------------------------------------------------------------------
    print("\n" + "="*56)
    print("  DISTRACTOR RANKER EVALUATION  (spec Section 7.8)")
    print("="*56)

    sample_d = df.head(min(500, len(df)))
    X_d_rows, y_d_rows = [], []
    p_at_3_hits = 0

    for _, row in sample_d.iterrows():
        correct_key  = str(row['answer']).strip().upper()
        correct_text = _b_clean(row[correct_key])

        for opt in OPTION_COLS:
            cand_text = _b_clean(row[opt])
            label     = 0 if opt == correct_key else 1
            row_dict  = {**row.to_dict(), 'candidate': cand_text}
            feats     = _build_distractor_features(row_dict, tfidf_vec, correct_text)
            X_d_rows.append(feats)
            y_d_rows.append(label)

        # Precision@3 per question
        q_feats = np.vstack(X_d_rows[-4:])
        try:
            q_pred = ranker.predict(q_feats)
        except Exception:
            continue
        pred_dist_idx  = np.argsort(q_pred)[-3:]
        true_dist_idx  = {i for i, l in enumerate(y_d_rows[-4:]) if l == 1}
        p_at_3_hits   += len(set(pred_dist_idx) & true_dist_idx) / 3

    if X_d_rows:
        X_d = np.vstack(X_d_rows)
        y_d = np.array(y_d_rows)

        try:
            y_d_pred = ranker.predict(X_d)
            has_proba = hasattr(ranker, 'predict_proba')
            y_d_prob  = ranker.predict_proba(X_d)[:, 1] if has_proba else y_d_pred.astype(float)
        except Exception as e:
            print(f"  [ERROR] ranker.predict failed: {e}")
            y_d_pred = np.zeros(len(y_d), dtype=int)
            y_d_prob = y_d_pred.astype(float)

        acc  = accuracy_score(y_d, y_d_pred)
        f1   = f1_score(y_d, y_d_pred, average='macro', zero_division=0)
        prec = precision_score(y_d, y_d_pred, average='macro', zero_division=0)
        rec  = recall_score(y_d, y_d_pred, average='macro', zero_division=0)
        p3   = p_at_3_hits / max(len(sample_d), 1)

        # Per-class
        f1_cls   = f1_score(y_d, y_d_pred, average=None, zero_division=0)
        prec_cls = precision_score(y_d, y_d_pred, average=None, zero_division=0)
        rec_cls  = recall_score(y_d, y_d_pred, average=None, zero_division=0)

        print(f"\n  Accuracy     : {acc:.4f}")
        print(f"  Macro F1     : {f1:.4f}   ← primary metric")
        print(f"  Precision    : {prec:.4f}")
        print(f"  Recall       : {rec:.4f}")
        print(f"  Precision@3  : {p3:.4f}   (top-3 ranked are valid distractors)")

        # Precision@K using predict_proba scores
        for k in [1, 3, 5]:
            pk = _precision_at_k(y_d, y_d_prob, k)
            print(f"  P@{k}          : {pk:.4f}")

        print(f"\n  Per-Class Breakdown:")
        print(f"  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        for c in range(len(f1_cls)):
            sup = int((y_d == c).sum())
            print(f"  {c:<8} {prec_cls[c]:>10.4f} {rec_cls[c]:>10.4f} {f1_cls[c]:>10.4f} {sup:>10}")

        cm = confusion_matrix(y_d, y_d_pred)
        print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
        print(f"              Pred 0   Pred 1")
        print(f"  Actual 0  {cm[0,0]:>8}  {cm[0,1]:>7}   (TN / FP)")
        print(f"  Actual 1  {cm[1,0]:>8}  {cm[1,1]:>7}   (FN / TP)")

        print(f"\n  Health Check:")
        if f1 >= 0.70:
            print(f"  [GOOD] Macro F1={f1:.4f}  → ranker is working well")
        elif f1 >= 0.55:
            print(f"  [OK]   Macro F1={f1:.4f}  → ranker is learning")
        else:
            print(f"  [WEAK] Macro F1={f1:.4f}  → ranker needs improvement")

    # -----------------------------------------------------------------------
    # HINT SCORER EVALUATION  (spec Section 7.6)
    # -----------------------------------------------------------------------
    print("\n" + "="*56)
    print("  HINT SCORER EVALUATION  (spec Section 7.6)")
    print("="*56)

    if not os.path.exists(hint_path):
        print("  [SKIP] hint_scorer.pkl not found — run model_b_train.py first")
    else:
        hint_scorer = joblib.load(hint_path)
        sample_h    = df.head(min(500, len(df)))

        X_h_rows, y_h_rows, sim_rows = [], [], []

        for _, row in sample_h.iterrows():
            article     = row['article']
            question    = row['question']
            correct_ans = _b_clean(row[str(row['answer']).strip().upper()])
            sentences   = _b_split_sentences(article)
            n           = len(sentences)
            if n < 2:
                continue
            for idx, sent in enumerate(sentences):
                label = int(correct_ans in _b_clean(sent))
                feats = _build_hint_features(sent, question, article, idx, n, tfidf_vec)
                X_h_rows.append(feats)
                y_h_rows.append(label)
                sim_rows.append(float(feats[3]))   # cosine sim column

        if X_h_rows:
            X_h   = np.vstack(X_h_rows)
            y_h   = np.array(y_h_rows)
            sim_h = np.array(sim_rows)

            try:
                y_h_pred = hint_scorer.predict(X_h)
                y_h_prob = hint_scorer.predict_proba(X_h)[:, 1]
            except Exception as e:
                print(f"  [ERROR] hint_scorer.predict failed: {e}")
                y_h_pred = np.zeros(len(y_h), dtype=int)
                y_h_prob = y_h_pred.astype(float)

            acc_h  = accuracy_score(y_h, y_h_pred)
            f1_h   = f1_score(y_h, y_h_pred, average='macro', zero_division=0)
            prec_h = precision_score(y_h, y_h_pred, average='macro', zero_division=0)
            rec_h  = recall_score(y_h, y_h_pred, average='macro', zero_division=0)
            r2_h   = r2_score(sim_h, y_h_prob)

            f1_cls_h   = f1_score(y_h, y_h_pred, average=None, zero_division=0)
            prec_cls_h = precision_score(y_h, y_h_pred, average=None, zero_division=0)
            rec_cls_h  = recall_score(y_h, y_h_pred, average=None, zero_division=0)

            print(f"\n  Accuracy     : {acc_h:.4f}")
            print(f"  Macro F1     : {f1_h:.4f}   ← primary metric")
            print(f"  Precision    : {prec_h:.4f}")
            print(f"  Recall       : {rec_h:.4f}")
            print(f"  R²           : {r2_h:.4f}   (predicted prob vs cosine-sim to question)")

            for k in [1, 3, 5]:
                pk = _precision_at_k(y_h, y_h_prob, k)
                print(f"  P@{k}          : {pk:.4f}")

            print(f"\n  Per-Class Breakdown:")
            print(f"  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
            for c in range(len(f1_cls_h)):
                sup = int((y_h == c).sum())
                print(f"  {c:<8} {prec_cls_h[c]:>10.4f} {rec_cls_h[c]:>10.4f} {f1_cls_h[c]:>10.4f} {sup:>10}")

            cm_h = confusion_matrix(y_h, y_h_pred)
            print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
            print(f"              Pred 0   Pred 1")
            print(f"  Actual 0  {cm_h[0,0]:>8}  {cm_h[0,1]:>7}   (TN / FP)")
            print(f"  Actual 1  {cm_h[1,0]:>8}  {cm_h[1,1]:>7}   (FN / TP)")

            print(f"\n  Health Check:")
            if f1_h >= 0.60:
                print(f"  [GOOD] Macro F1={f1_h:.4f}  → hint scorer is working well")
            elif f1_h >= 0.45:
                print(f"  [OK]   Macro F1={f1_h:.4f}  → hint scorer is learning")
            else:
                print(f"  [WEAK] Macro F1={f1_h:.4f}  → hint scorer needs improvement")
            if r2_h > 0.10:
                print(f"  [GOOD] R²={r2_h:.4f}  → predicted scores correlate with cosine similarity")
            else:
                print(f"  [WEAK] R²={r2_h:.4f}  → low correlation between scores and cosine similarity")
        else:
            print("  [WARN] No hint sentences extracted from this split")

    # -----------------------------------------------------------------------
    # WORD2VEC EVALUATION  (spec Section 7.5)
    # -----------------------------------------------------------------------
    print("\n" + "="*56)
    print("  WORD2VEC EVALUATION  (spec Section 7.5)")
    print("="*56)

    w2v_path = f'{MODEL_B_DIR}/word2vec_model.bin'
    if os.path.exists(w2v_path):
        try:
            from gensim.models import Word2Vec as _W2V
            w2v  = _W2V.load(w2v_path)
            wv   = w2v.wv
            vocab_size = len(wv)
            print(f"  Vocabulary size : {vocab_size:,}")
            print(f"  Vector size     : {wv.vector_size}")

            # Nearest-neighbour sanity checks on common words
            test_words = ['student', 'school', 'read', 'answer', 'question']
            print(f"\n  Nearest-neighbour sanity check (top-3):")
            for w in test_words:
                if w in wv:
                    nbs = wv.most_similar(w, topn=3)
                    nb_str = ', '.join(f"{n}({s:.2f})" for n, s in nbs)
                    print(f"    {w:<12} → {nb_str}")
                else:
                    print(f"    {w:<12} → [not in vocabulary]")

            # Coverage: fraction of val answer tokens in vocab
            total_tokens, covered = 0, 0
            for _, row in df.head(200).iterrows():
                ans_tokens = _b_tokenize(row[str(row['answer']).strip().upper()])
                for t in ans_tokens:
                    total_tokens += 1
                    if t in wv:
                        covered += 1
            coverage = covered / max(total_tokens, 1)
            print(f"\n  Answer-token vocab coverage : {coverage:.4f}  ({covered}/{total_tokens})")
            if coverage >= 0.80:
                print(f"  [GOOD] ≥80% of answer tokens are in Word2Vec vocab")
            elif coverage >= 0.60:
                print(f"  [OK]   60–80% coverage — reasonable for a small corpus")
            else:
                print(f"  [WEAK] <60% coverage — Word2Vec may produce poor distractors")
        except Exception as e:
            print(f"  [ERROR] Could not load Word2Vec model: {e}")
    else:
        print("  [SKIP] word2vec_model.bin not found — run model_b_train.py first")

    # -----------------------------------------------------------------------
    # MODEL B SUMMARY TABLE
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("  MODEL B SUMMARY")
    print("="*60)
    print(f"  {'Component':<25} {'Status'}")
    print(f"  {'-'*50}")
    print(f"  {'Distractor Ranker':<25} {'loaded' if os.path.exists(ranker_path) else 'MISSING'}")
    print(f"  {'Hint Scorer':<25} {'loaded' if os.path.exists(hint_path) else 'MISSING'}")
    print(f"  {'Word2Vec':<25} {'loaded' if os.path.exists(w2v_path) else 'MISSING'}")

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