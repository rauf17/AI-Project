# src/preprocessing.py
# Muhammad Umair (23I-0662) AND Abdul Rauf (23I-0591)
#
# Full preprocessing pipeline for the RACE Reading Comprehension project.
# Implements ALL steps from the project specification:
#   1. Load raw CSVs
#   2. Text cleaning
#   3. Build 4x expanded binary label matrix (y)
#   4. One-Hot Encoding (PRIMARY representation)
#   5. TF-IDF Vectorization (OPTIONAL)
#   6. Cosine Similarity Features (handcrafted)
#   7. Handcrafted Lexical Features
#   8. Save all processed outputs
#
# Run: python src/preprocessing.py

import os
import re
import sys
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Directory Setup
# ---------------------------------------------------------------------------

RAW_DIR       = './data/raw'
PROCESSED_DIR = './data/processed'
MODEL_A_DIR   = './models/model_a/traditional'

for d in [PROCESSED_DIR, MODEL_A_DIR]:
    os.makedirs(d, exist_ok=True)

OPTION_COLS = ['A', 'B', 'C', 'D']

# ---------------------------------------------------------------------------
# Step 1 — Load Raw Data
# ---------------------------------------------------------------------------

def load_data():
    """Load raw RACE CSV files and drop rows with any missing values."""
    print("[1/7] Loading raw datasets...")
    train_df = pd.read_csv(f'{RAW_DIR}/train.csv').dropna()
    val_df   = pd.read_csv(f'{RAW_DIR}/val.csv').dropna()
    test_df  = pd.read_csv(f'{RAW_DIR}/test.csv').dropna()

    print(f"  Train: {train_df.shape[0]:,} rows | Val: {val_df.shape[0]:,} rows | Test: {test_df.shape[0]:,} rows")
    return train_df, val_df, test_df

# ---------------------------------------------------------------------------
# Step 2 — Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)   # strip punctuation (keeps alphanumeric + spaces)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_text to article, question, and all option columns in-place."""
    df = df.copy()
    for col in ['article', 'question'] + OPTION_COLS:
        df[col] = df[col].apply(clean_text)
    return df

# ---------------------------------------------------------------------------
# Step 3 — Build 4x Expanded Label Matrix
# ---------------------------------------------------------------------------

def expand_to_four_rows(df: pd.DataFrame):
    """
    For every passage-question pair, create 4 rows (one per option A/B/C/D).
    The 'combined' text = article + article + question + option  (article doubled
    to give it more weight over the short option string, per spec Section 5.1.4c).
    Label = 1 if this option is the correct answer, else 0.
    Expected: y.mean() ≈ 0.25
    """
    rows = []
    for _, r in df.iterrows():
        correct = str(r['answer']).strip().upper()
        for opt in OPTION_COLS:
            combined = (
                r['article'] + ' ' +
                r['article'] + ' ' +
                r['question'] + ' ' +
                r[opt]
            )
            label = 1 if opt == correct else 0
            rows.append({'combined': combined, 'label': label})
    expanded = pd.DataFrame(rows)
    return expanded


def build_label_matrix(train_df, val_df, test_df):
    """Expand each split to 4× rows and return (texts, labels) per split."""
    print("[3/7] Expanding to 4-row label matrices...")
    train_exp = expand_to_four_rows(train_df)
    val_exp   = expand_to_four_rows(val_df)
    test_exp  = expand_to_four_rows(test_df)

    print(f"  Train expanded: {len(train_exp):,} rows | y.mean = {train_exp['label'].mean():.4f} (expected ≈ 0.25)")
    print(f"  Val   expanded: {len(val_exp):,} rows")
    print(f"  Test  expanded: {len(test_exp):,} rows")

    return train_exp, val_exp, test_exp

# ---------------------------------------------------------------------------
# Step 4 — One-Hot Encoding (Primary Feature Representation)
# ---------------------------------------------------------------------------

def build_onehot_features(train_exp, val_exp, test_exp):
    """
    Fit CountVectorizer on training corpus → binarize counts → sparse One-Hot matrix.
    IMPORTANT: fit_transform only on train; transform on val/test to avoid data leakage.
    """
    print("[4/7] Building One-Hot features (primary)...")

    # 4a. Fit vocabulary on training data
    vocab_vectorizer = CountVectorizer(
        max_features=10000,
        stop_words='english',
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 1)
    )
    X_train_counts = vocab_vectorizer.fit_transform(train_exp['combined'])
    X_val_counts   = vocab_vectorizer.transform(val_exp['combined'])
    X_test_counts  = vocab_vectorizer.transform(test_exp['combined'])

    # 4b. Binarize: convert counts to binary (0/1) presence flags
    binarizer = Binarizer()
    X_train_onehot = binarizer.fit_transform(X_train_counts)
    X_val_onehot   = binarizer.transform(X_val_counts)
    X_test_onehot  = binarizer.transform(X_test_counts)

    print(f"  One-Hot shape: train={X_train_onehot.shape}, val={X_val_onehot.shape}, test={X_test_onehot.shape}")

    # Save fitted encoder
    joblib.dump(vocab_vectorizer, f'{MODEL_A_DIR}/onehot_encoder.pkl')
    print(f"  Saved onehot_encoder.pkl")

    return X_train_onehot, X_val_onehot, X_test_onehot, vocab_vectorizer

# ---------------------------------------------------------------------------
# Step 5 — TF-IDF Vectorization (Optional)
# ---------------------------------------------------------------------------

def build_tfidf_features(train_exp, val_exp, test_exp):
    """Optional TF-IDF representation on the same combined text."""
    print("[5/7] Building TF-IDF features (optional)...")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        sublinear_tf=True,       # log(1+TF) — dampens high-frequency terms
        ngram_range=(1, 2),      # unigrams + bigrams
        min_df=2,
        max_df=0.95,
        norm='l2'                # required for cosine similarity
    )

    X_train_tfidf = vectorizer.fit_transform(train_exp['combined'])
    X_val_tfidf   = vectorizer.transform(val_exp['combined'])
    X_test_tfidf  = vectorizer.transform(test_exp['combined'])

    joblib.dump(vectorizer, f'{MODEL_A_DIR}/tfidf_vectorizer.pkl')
    sp.save_npz(f'{PROCESSED_DIR}/X_train_tfidf.npz', X_train_tfidf)

    print(f"  TF-IDF shape: train={X_train_tfidf.shape}")
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer

# ---------------------------------------------------------------------------
# Step 6 — Cosine Similarity Features (Handcrafted)
# ---------------------------------------------------------------------------

def _safe_cosine(vec_a, vec_b):
    """Return scalar cosine similarity between two sparse row vectors."""
    sim = cosine_similarity(vec_a, vec_b)
    return float(sim[0, 0])


def build_cosine_features(df_orig: pd.DataFrame, vectorizer) -> np.ndarray:
    """
    For every row in the *original* (unexpanded) dataframe, compute 6 cosine /
    lexical features for each of the 4 options, yielding shape (4*N, 6).

    Features per option:
        F1  cosine(article_vec, option_vec)
        F2  cosine(question_vec, option_vec)
        F3  cosine(article_vec, question_vec)
        F4  character-level overlap ratio (option ∩ answer chars)
        F5  word length ratio len(option) / max(len(article), 1)
        F6  passage frequency of option keywords (fraction of option
            tokens that appear in article)
    """
    feature_rows = []

    for _, r in df_orig.iterrows():
        art_vec = vectorizer.transform([r['article']])
        q_vec   = vectorizer.transform([r['question']])
        art_q_sim = _safe_cosine(art_vec, q_vec)

        art_tokens  = set(r['article'].split())
        art_len     = max(len(r['article'].split()), 1)

        for opt in OPTION_COLS:
            opt_text = r[opt]
            opt_vec  = vectorizer.transform([opt_text])

            f1 = _safe_cosine(art_vec, opt_vec)
            f2 = _safe_cosine(q_vec, opt_vec)
            f3 = art_q_sim
            # F4: character-level overlap (shared chars / max length)
            chars_opt = set(opt_text)
            chars_art = set(r['article'])
            f4 = len(chars_opt & chars_art) / max(len(chars_opt | chars_art), 1)
            # F5: word length ratio
            f5 = len(opt_text.split()) / art_len
            # F6: fraction of option tokens found in article
            opt_tokens = set(opt_text.split())
            f6 = len(opt_tokens & art_tokens) / max(len(opt_tokens), 1)

            feature_rows.append([f1, f2, f3, f4, f5, f6])

    return np.array(feature_rows, dtype=np.float32)


def append_cosine_features(X_sparse, cosine_arr: np.ndarray):
    """Horizontally stack sparse One-Hot matrix with dense cosine features."""
    cosine_sparse = sp.csr_matrix(cosine_arr)
    return sp.hstack([X_sparse, cosine_sparse], format='csr')

# ---------------------------------------------------------------------------
# Step 7 — Handcrafted Lexical Features
# ---------------------------------------------------------------------------

def jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def count_named_entities(text: str) -> int:
    """Rough NE count: capitalized words (after first token in sentence)."""
    words = text.split()
    return sum(1 for w in words[1:] if w and w[0].isupper())


def build_lexical_features(df_orig: pd.DataFrame) -> np.ndarray:
    """
    7 handcrafted lexical features per option row (4 rows per original row):
        L1  keyword overlap count (question tokens ∩ option tokens)
        L2  option length (word count)
        L3  article length (sentence count)
        L4  position of first matching sentence (normalized 0–1)
        L5  Jaccard similarity between question tokens and option tokens
        L6  number of named entities in option (capitalized word count)
        L7  keyword overlap count (article tokens ∩ option tokens)
    """
    rows = []

    for _, r in df_orig.iterrows():
        q_tokens  = set(r['question'].split())
        art_sents = [s.strip() for s in r['article'].split('.') if s.strip()]
        art_len_sents = max(len(art_sents), 1)

        for opt in OPTION_COLS:
            opt_tokens  = set(r[opt].split())
            art_tokens  = set(r['article'].split())

            l1 = len(q_tokens & opt_tokens)
            l2 = len(r[opt].split())
            l3 = art_len_sents
            # L4: position of first sentence containing an option token
            pos = art_len_sents  # default: not found → end of article
            for i, sent in enumerate(art_sents):
                if opt_tokens & set(sent.split()):
                    pos = i
                    break
            l4 = pos / art_len_sents
            l5 = jaccard(q_tokens, opt_tokens)
            l6 = count_named_entities(r[opt])
            l7 = len(art_tokens & opt_tokens)

            rows.append([l1, l2, l3, l4, l5, l6, l7])

    return np.array(rows, dtype=np.float32)

# ---------------------------------------------------------------------------
# Step 8 — Save All Processed Outputs
# ---------------------------------------------------------------------------

def save_labels(train_exp, val_exp, test_exp):
    np.save(f'{PROCESSED_DIR}/y_train.npy', train_exp['label'].values)
    np.save(f'{PROCESSED_DIR}/y_val.npy',   val_exp['label'].values)
    np.save(f'{PROCESSED_DIR}/y_test.npy',  test_exp['label'].values)
    print("  Saved y_train / y_val / y_test")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  RACE Preprocessing Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    train_df, val_df, test_df = load_data()

    # ------------------------------------------------------------------
    # 2. Clean text
    # ------------------------------------------------------------------
    print("[2/7] Cleaning text...")
    train_df = clean_dataframe(train_df)
    val_df   = clean_dataframe(val_df)
    test_df  = clean_dataframe(test_df)
    print("  Text cleaning complete.")

    # ------------------------------------------------------------------
    # 3. Build 4x expanded label matrices
    # ------------------------------------------------------------------
    train_exp, val_exp, test_exp = build_label_matrix(train_df, val_df, test_df)
    save_labels(train_exp, val_exp, test_exp)

    # ------------------------------------------------------------------
    # 4. One-Hot Encoding (PRIMARY)
    # ------------------------------------------------------------------
    X_train_oh, X_val_oh, X_test_oh, vocab_vec = build_onehot_features(
        train_exp, val_exp, test_exp
    )

    # ------------------------------------------------------------------
    # 5. TF-IDF (OPTIONAL)
    # ------------------------------------------------------------------
    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vec = build_tfidf_features(
        train_exp, val_exp, test_exp
    )

    # ------------------------------------------------------------------
    # 6. Cosine Similarity Features
    #    Computed on original (unexpanded) dataframes using TF-IDF vectorizer
    #    (higher quality similarity than raw One-Hot counts)
    # ------------------------------------------------------------------
    print("[6/7] Building cosine similarity features...")
    print("  (This may take a few minutes on the full RACE dataset)")

    cos_train = build_cosine_features(train_df, tfidf_vec)
    cos_val   = build_cosine_features(val_df,   tfidf_vec)
    cos_test  = build_cosine_features(test_df,  tfidf_vec)

    # Append cosine features to One-Hot matrices
    X_train_combined = append_cosine_features(X_train_oh, cos_train)
    X_val_combined   = append_cosine_features(X_val_oh,   cos_val)
    X_test_combined  = append_cosine_features(X_test_oh,  cos_test)

    # Save all cosine features (stacked across splits for convenience)
    all_cos = np.vstack([cos_train, cos_val, cos_test])
    sp.save_npz(f'{PROCESSED_DIR}/cosine_features.npz', sp.csr_matrix(all_cos))

    # ------------------------------------------------------------------
    # 7. Handcrafted Lexical Features
    # ------------------------------------------------------------------
    print("[7/7] Building handcrafted lexical features...")
    lex_train = build_lexical_features(train_df)
    lex_val   = build_lexical_features(val_df)
    lex_test  = build_lexical_features(test_df)

    # Append lexical features to combined matrices
    X_train_final = sp.hstack([X_train_combined, sp.csr_matrix(lex_train)], format='csr')
    X_val_final   = sp.hstack([X_val_combined,   sp.csr_matrix(lex_val)],   format='csr')
    X_test_final  = sp.hstack([X_test_combined,  sp.csr_matrix(lex_test)],  format='csr')

    # ------------------------------------------------------------------
    # 8. Save final One-Hot + cosine + lexical matrices
    # ------------------------------------------------------------------
    print("[8/8] Saving processed feature matrices...")
    sp.save_npz(f'{PROCESSED_DIR}/X_train_onehot.npz', X_train_final)
    sp.save_npz(f'{PROCESSED_DIR}/X_val_onehot.npz',   X_val_final)
    sp.save_npz(f'{PROCESSED_DIR}/X_test_onehot.npz',  X_test_final)

    print()
    print("=" * 60)
    print("  Preprocessing complete! Summary:")
    print("=" * 60)
    print(f"  X_train shape : {X_train_final.shape}  (One-Hot + cosine + lexical)")
    print(f"  X_val shape   : {X_val_final.shape}")
    print(f"  X_test shape  : {X_test_final.shape}")
    print(f"  y_train mean  : {np.load(PROCESSED_DIR+'/y_train.npy').mean():.4f}  (expected ≈ 0.25)")
    print()
    print("  Saved artefacts:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        path = os.path.join(PROCESSED_DIR, f)
        size_mb = os.path.getsize(path) / 1e6
        print(f"    {PROCESSED_DIR}/{f}  ({size_mb:.1f} MB)")
    for f in sorted(os.listdir(MODEL_A_DIR)):
        print(f"    {MODEL_A_DIR}/{f}")


if __name__ == '__main__':
    main()