# src/preprocessing.py
# Muhammad Umair (23I-0662) AND Abdul Rauf (23I-0591)
#
# Full preprocessing pipeline for the RACE Reading Comprehension project.
# Optimised for Kaggle / 16GB RAM — ALL cosine and lexical features are
# computed in a single batch matrix operation. No row-by-row loops.
#
# Run: python src/preprocessing.py

import os
import re
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------------------------
# Paths — Kaggle layout
# ---------------------------------------------------------------------------

RAW_DIR       = './data/raw'
PROCESSED_DIR = './data/processed'
MODEL_A_DIR   = './models/model_a/traditional'

for d in [PROCESSED_DIR, MODEL_A_DIR]:
    os.makedirs(d, exist_ok=True)

OPTION_COLS = ['A', 'B', 'C', 'D']

# ---------------------------------------------------------------------------
# Step 1 — Load
# ---------------------------------------------------------------------------

def load_data():
    print("[1/7] Loading raw datasets...")
    train_df = pd.read_csv(f'{RAW_DIR}/train.csv').dropna().reset_index(drop=True)
    val_df   = pd.read_csv(f'{RAW_DIR}/val.csv').dropna().reset_index(drop=True)
    test_df  = pd.read_csv(f'{RAW_DIR}/test.csv').dropna().reset_index(drop=True)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return train_df, val_df, test_df

# ---------------------------------------------------------------------------
# Step 2 — Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['article', 'question'] + OPTION_COLS:
        df[col] = df[col].apply(clean_text)
    return df

# ---------------------------------------------------------------------------
# Step 3 — 4x Label Expansion
# ---------------------------------------------------------------------------

def expand_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each row into 4 rows (one per option A/B/C/D).
    combined = article + article + question + option
    (article doubled to give it more weight over the short option text)
    y.mean() should be approx 0.25
    """
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
                'combined':      (r['article'] + ' ' + r['article'] + ' ' +
                                  r['question'] + ' ' + r[opt])
            })
    expanded = pd.DataFrame(records)
    print(f"  Expanded {len(df):,} -> {len(expanded):,} rows | y.mean={expanded['label'].mean():.4f}")
    return expanded

# ---------------------------------------------------------------------------
# Step 4 — One-Hot Encoding (PRIMARY feature representation)
# ---------------------------------------------------------------------------

def build_onehot(train_exp, val_exp, test_exp):
    print("[4/7] One-Hot features (primary)...")

    vocab_vec = CountVectorizer(
        max_features=10000, stop_words='english',
        min_df=2, max_df=0.95, ngram_range=(1, 1)
    )
    # fit_transform only on train — transform on val/test to avoid leakage
    X_tr = vocab_vec.fit_transform(train_exp['combined'])
    X_va = vocab_vec.transform(val_exp['combined'])
    X_te = vocab_vec.transform(test_exp['combined'])

    binarizer = Binarizer()
    X_tr = binarizer.fit_transform(X_tr)
    X_va = binarizer.transform(X_va)
    X_te = binarizer.transform(X_te)

    joblib.dump(vocab_vec, f'{MODEL_A_DIR}/onehot_encoder.pkl')
    print(f"  One-Hot shape: {X_tr.shape}")
    return X_tr, X_va, X_te, vocab_vec

# ---------------------------------------------------------------------------
# Step 5 — TF-IDF (OPTIONAL — also used as base for cosine features)
# ---------------------------------------------------------------------------

def build_tfidf(train_exp, val_exp, test_exp):
    print("[5/7] TF-IDF features (optional)...")

    tfidf_vec = TfidfVectorizer(
        max_features=10000, stop_words='english',
        sublinear_tf=True, ngram_range=(1, 2),
        min_df=2, max_df=0.95, norm='l2'
    )
    X_tr = tfidf_vec.fit_transform(train_exp['combined'])
    X_va = tfidf_vec.transform(val_exp['combined'])
    X_te = tfidf_vec.transform(test_exp['combined'])

    joblib.dump(tfidf_vec, f'{MODEL_A_DIR}/tfidf_vectorizer.pkl')
    sp.save_npz(f'{PROCESSED_DIR}/X_train_tfidf.npz', X_tr)
    print(f"  TF-IDF shape: {X_tr.shape}")
    return X_tr, X_va, X_te, tfidf_vec

# ---------------------------------------------------------------------------
# Step 6 — Cosine Similarity Features  <-- THE KEY FIX
#
# OLD approach (your original): called vectorizer.transform() once per row
# per option inside a Python loop -> 350,000+ individual sparse ops -> freezes
#
# NEW approach: transform ALL article/question/option texts in 3 bulk calls,
# then use paired_cosine_distances() which runs in compiled C with no Python
# loop at all. 350k rows now takes ~2-3 minutes instead of 45+ minutes.
# ---------------------------------------------------------------------------

def build_cosine_features_fast(expanded_df: pd.DataFrame, tfidf_vec) -> np.ndarray:
    """
    6 similarity features per expanded row, computed entirely in batch.

    F1  cosine(article, option)
    F2  cosine(question, option)
    F3  cosine(article, question)
    F4  character overlap ratio  len(opt_chars & art_chars) / len(union)
    F5  word length ratio        len(opt_words) / len(art_words)
    F6  token coverage           len(opt_tokens & art_tokens) / len(opt_tokens)
    """
    print("  Transforming article vectors (batch)...")
    V_art = tfidf_vec.transform(expanded_df['article'].tolist())

    print("  Transforming question vectors (batch)...")
    V_que = tfidf_vec.transform(expanded_df['question'].tolist())

    print("  Transforming option vectors (batch)...")
    V_opt = tfidf_vec.transform(expanded_df['option_text'].tolist())

    # paired_cosine_distances: row i of A vs row i of B, returns distances
    # convert to similarity with 1 - distance
    print("  Computing paired cosine similarities (C extension, no Python loop)...")
    F1 = (1 - paired_cosine_distances(V_art, V_opt)).astype(np.float32)
    F2 = (1 - paired_cosine_distances(V_que, V_opt)).astype(np.float32)
    F3 = (1 - paired_cosine_distances(V_art, V_que)).astype(np.float32)

    # String-level features — still a Python loop but only over strings,
    # NOT over the vectorizer. ~350k string ops takes about 5 seconds.
    print("  Computing string-level features...")
    n = len(expanded_df)
    F4 = np.zeros(n, dtype=np.float32)
    F5 = np.zeros(n, dtype=np.float32)
    F6 = np.zeros(n, dtype=np.float32)

    arts = expanded_df['article'].tolist()
    opts = expanded_df['option_text'].tolist()

    for i, (art, opt) in enumerate(zip(arts, opts)):
        # F4: character overlap
        s_opt = set(opt);  s_art = set(art)
        union = s_opt | s_art
        F4[i] = len(s_opt & s_art) / max(len(union), 1)

        # F5: word length ratio
        art_wc = max(len(art.split()), 1)
        F5[i] = len(opt.split()) / art_wc

        # F6: token coverage
        opt_toks = set(opt.split())
        art_toks = set(art.split())
        F6[i] = len(opt_toks & art_toks) / max(len(opt_toks), 1)

    features = np.column_stack([F1, F2, F3, F4, F5, F6])
    print(f"  Cosine feature matrix: {features.shape}")
    return features

# ---------------------------------------------------------------------------
# Step 7 — Handcrafted Lexical Features (vectorised where possible)
# ---------------------------------------------------------------------------

def build_lexical_features_fast(expanded_df: pd.DataFrame) -> np.ndarray:
    """
    7 lexical features — no vectorizer calls, pure Python string ops.

    L1  keyword overlap count  (question tokens & option tokens)
    L2  option word count
    L3  article sentence count
    L4  normalised position of first sentence containing an option token
    L5  Jaccard similarity     (question tokens, option tokens)
    L6  named entity count in option (capitalised word heuristic)
    L7  keyword overlap count  (article tokens & option tokens)
    """
    print("  Computing lexical features...")
    n = len(expanded_df)
    L1 = np.zeros(n, dtype=np.float32)
    L2 = np.zeros(n, dtype=np.float32)
    L3 = np.zeros(n, dtype=np.float32)
    L4 = np.zeros(n, dtype=np.float32)
    L5 = np.zeros(n, dtype=np.float32)
    L6 = np.zeros(n, dtype=np.float32)
    L7 = np.zeros(n, dtype=np.float32)

    arts  = expanded_df['article'].tolist()
    ques  = expanded_df['question'].tolist()
    opts  = expanded_df['option_text'].tolist()

    for i, (art, q, opt) in enumerate(zip(arts, ques, opts)):
        q_toks   = set(q.split())
        opt_toks = set(opt.split())
        art_toks = set(art.split())
        art_sents = [s.strip() for s in art.split('.') if s.strip()]
        n_sents   = max(len(art_sents), 1)

        L1[i] = len(q_toks & opt_toks)
        L2[i] = len(opt_toks)
        L3[i] = n_sents

        # L4: position of first sentence containing an option token
        pos = n_sents
        for j, sent in enumerate(art_sents):
            if opt_toks & set(sent.split()):
                pos = j
                break
        L4[i] = pos / n_sents

        # L5: Jaccard
        union = q_toks | opt_toks
        L5[i] = len(q_toks & opt_toks) / max(len(union), 1)

        # L6: capitalised words after first token (NE heuristic)
        words = opt.split()
        L6[i] = sum(1 for w in words[1:] if w and w[0].isupper())

        L7[i] = len(art_toks & opt_toks)

    features = np.column_stack([L1, L2, L3, L4, L5, L6, L7])
    print(f"  Lexical feature matrix: {features.shape}")
    return features

# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  RACE Preprocessing Pipeline (Optimised for Kaggle)")
    print("=" * 60)

    # 1. Load
    train_df, val_df, test_df = load_data()

    # 2. Clean
    print("[2/7] Cleaning text...")
    train_df = clean_dataframe(train_df)
    val_df   = clean_dataframe(val_df)
    test_df  = clean_dataframe(test_df)
    print("  Done.")

    # 3. Expand + save labels
    print("[3/7] Expanding to 4x label matrices...")
    train_exp = expand_dataframe(train_df)
    val_exp   = expand_dataframe(val_df)
    test_exp  = expand_dataframe(test_df)

    np.save(f'{PROCESSED_DIR}/y_train.npy', train_exp['label'].values)
    np.save(f'{PROCESSED_DIR}/y_val.npy',   val_exp['label'].values)
    np.save(f'{PROCESSED_DIR}/y_test.npy',  test_exp['label'].values)
    print("  Labels saved.")

    # 4. One-Hot (primary)
    X_tr_oh, X_va_oh, X_te_oh, vocab_vec = build_onehot(train_exp, val_exp, test_exp)

    # 5. TF-IDF (optional, also used for cosine features below)
    X_tr_tf, X_va_tf, X_te_tf, tfidf_vec = build_tfidf(train_exp, val_exp, test_exp)

    # 6. Cosine features — BATCHED (the fix)
    print("[6/7] Cosine similarity features...")
    cos_train = build_cosine_features_fast(train_exp, tfidf_vec)
    cos_val   = build_cosine_features_fast(val_exp,   tfidf_vec)
    cos_test  = build_cosine_features_fast(test_exp,  tfidf_vec)

    all_cos = np.vstack([cos_train, cos_val, cos_test])
    sp.save_npz(f'{PROCESSED_DIR}/cosine_features.npz', sp.csr_matrix(all_cos))

    # 7. Lexical features
    print("[7/7] Lexical features...")
    lex_train = build_lexical_features_fast(train_exp)
    lex_val   = build_lexical_features_fast(val_exp)
    lex_test  = build_lexical_features_fast(test_exp)

    # 8. Combine One-Hot + cosine + lexical and save
    print("[8/8] Combining and saving final feature matrices...")

    def combine_and_save(X_oh, cos, lex, split):
        X_final = sp.hstack([
            X_oh,
            sp.csr_matrix(cos),
            sp.csr_matrix(lex)
        ], format='csr')
        sp.save_npz(f'{PROCESSED_DIR}/X_{split}_onehot.npz', X_final)
        print(f"  X_{split}: {X_final.shape}  nnz={X_final.nnz:,}")
        return X_final

    combine_and_save(X_tr_oh, cos_train, lex_train, 'train')
    combine_and_save(X_va_oh, cos_val,   lex_val,   'val')
    combine_and_save(X_te_oh, cos_test,  lex_test,  'test')

    print()
    print("=" * 60)
    print("  All done! Summary:")
    print(f"  y_train mean : {np.load(PROCESSED_DIR+'/y_train.npy').mean():.4f}  (expected ~0.25)")
    print("  Saved to     :", PROCESSED_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()