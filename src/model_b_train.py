# =============================================================================
# Model B — Distractor & Hint Generator — FULL TRAINING PIPELINE
# Muhammad Umair (23I-0662) AND Abdul Rauf (23I-0591)
#
# This script trains and saves:
#   1. distractor_ranker.pkl  — LR classifier to rank/score distractor candidates
#   2. hint_scorer.pkl        — LR classifier to score hint sentences
#   3. word2vec_model.bin     — Word2Vec model fine-tuned on RACE corpus
#
# Covers (per spec):
#   - Supervised Distractor Ranking Pipeline (Steps 1–3)
#   - One-Hot + Cosine Similarity distractor ranking
#   - Word2Vec nearest-neighbour distractors (alternative)
#   - Extractive + ML-Scored Hint Generation
#   - Full evaluation: Precision, Recall, F1, Accuracy, Confusion Matrix,
#     Precision@K, R² for hint scorer
# =============================================================================

import os
import re
import math
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp

from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, r2_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR       = "./data/raw"
PROC_DIR       = "./data/processed"
MODEL_A_DIR    = "./models/model_a/traditional"
MODEL_B_DIR    = "./models/model_b/traditional"

os.makedirs(PROC_DIR,    exist_ok=True)
os.makedirs(MODEL_B_DIR, exist_ok=True)

STOPWORDS = {
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

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return [w for w in clean_text(text).split() if w not in STOPWORDS and len(w) > 1]


def split_sentences(article: str):
    """Simple rule-based sentence splitter."""
    raw = re.split(r"(?<=[.!?])\s+", article.replace("\n", " "))
    return [s.strip() for s in raw if len(s.strip()) > 10]


def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def keyword_overlap(tokens_a, tokens_b):
    return len(set(tokens_a) & set(tokens_b))

# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def load_data():
    print("[1/7] Loading RACE CSVs …")
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    val_df   = pd.read_csv(f"{DATA_DIR}/val.csv")
    test_df  = pd.read_csv(f"{DATA_DIR}/test.csv")

    for df in (train_df, val_df, test_df):
        for col in ["article", "question", "A", "B", "C", "D"]:
            df[col] = df[col].astype(str).fillna("")

    print(f"    Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    return train_df, val_df, test_df

# ---------------------------------------------------------------------------
# 2. WORD2VEC TRAINING  (Section 7.5)
# ---------------------------------------------------------------------------

def train_word2vec(train_df, val_df):
    """
    Fine-tune a Word2Vec model on the RACE corpus so that nearest-neighbours
    can be used as distractor candidates at inference time.
    """
    print("[2/7] Training Word2Vec on RACE corpus …")
    sentences = []
    for df in (train_df, val_df):
        for _, row in df.iterrows():
            sentences.append(tokenize(row["article"]))
            sentences.append(tokenize(row["question"]))
            for opt in ["A", "B", "C", "D"]:
                sentences.append(tokenize(row[opt]))

    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        epochs=5,
        seed=42,
    )
    out_path = f"{MODEL_B_DIR}/word2vec_model.bin"
    model.save(out_path)
    print(f"    Word2Vec saved → {out_path}  (vocab size: {len(model.wv):,})")
    return model

# ---------------------------------------------------------------------------
# 3. DISTRACTOR RANKING PIPELINE  (Section 7.3 + 7.4)
# ---------------------------------------------------------------------------

def build_distractor_features(row, vectorizer, answer_col=None):
    """
    For a single (article, question, option, correct_answer) tuple compute
    the feature vector described in Section 7.3 Step 2.

    Returns a 1-D numpy array of 7 floats.
    """
    article       = clean_text(row["article"])
    correct_ans   = clean_text(row[row["answer"]]) if answer_col is None else clean_text(answer_col)
    candidate     = clean_text(row.get("candidate", ""))

    if not candidate:
        return np.zeros(7)

    # TF-IDF / One-Hot cosine similarities
    try:
        ans_vec  = vectorizer.transform([correct_ans])
        cand_vec = vectorizer.transform([candidate])
        art_vec  = vectorizer.transform([article])

        sim_ans_cand  = float(cosine_similarity(ans_vec, cand_vec)[0, 0])
        sim_art_cand  = float(cosine_similarity(art_vec, cand_vec)[0, 0])
    except Exception:
        sim_ans_cand = sim_art_cand = 0.0

    # Character-level overlap (normalised)
    chars_a = set(correct_ans)
    chars_c = set(candidate)
    char_overlap = len(chars_a & chars_c) / max(len(chars_a | chars_c), 1)

    # Passage frequency of candidate tokens
    art_tokens  = tokenize(row["article"])
    cand_tokens = tokenize(candidate)
    art_freq    = Counter(art_tokens)
    cand_freq   = sum(art_freq.get(t, 0) for t in cand_tokens) / max(len(cand_tokens), 1)

    # Position of first occurrence (normalised)
    article_lower = row["article"].lower()
    cand_lower    = candidate.lower().split()[0] if cand_tokens else ""
    pos = article_lower.find(cand_lower)
    norm_pos = (pos / max(len(article_lower), 1)) if pos >= 0 else 1.0

    # Word-length difference from correct answer (normalised)
    len_diff = abs(len(cand_tokens) - len(tokenize(correct_ans))) / max(len(tokenize(correct_ans)), 1)

    # Jaccard similarity between candidate and correct answer token sets
    jac = jaccard(cand_tokens, tokenize(correct_ans))

    return np.array([
        sim_ans_cand,   # 0: One-Hot/TF-IDF cosine(answer, candidate)
        sim_art_cand,   # 1: cosine(article, candidate)
        char_overlap,   # 2: character-level match score
        cand_freq,      # 3: passage frequency of candidate
        norm_pos,       # 4: first position in passage (normalised)
        len_diff,       # 5: word-length difference from answer
        jac,            # 6: Jaccard(candidate tokens, answer tokens)
    ], dtype=np.float32)


def build_distractor_dataset(df, vectorizer, max_rows=30_000):
    """
    For each RACE row treat options B/C/D as negative candidates and the
    correct option as positive (label=1).

    Label = 1  ↔  this candidate IS one of the original distractor options
    Label = 0  ↔  this candidate is the correct answer (should NOT be distractor)

    We follow the spec: "Label = 1 if the candidate is one of the original
    dataset's distractor options, else 0."
    """
    print("    Building distractor feature matrix …")
    X_rows, y_rows = [], []
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    for i, row in df.iterrows():
        if i >= max_rows:
            break
        correct_key = row["answer"]           # e.g. "A"
        correct_text = clean_text(row[correct_key])

        for opt in ["A", "B", "C", "D"]:
            candidate_text = clean_text(row[opt])
            # label=1 if this option is a distractor (not the correct answer)
            label = 0 if opt == correct_key else 1

            tmp = row.to_dict()
            tmp["candidate"] = candidate_text
            feats = build_distractor_features(tmp, vectorizer, answer_col=correct_text)
            X_rows.append(feats)
            y_rows.append(label)

    X = np.vstack(X_rows)
    y = np.array(y_rows)
    print(f"    Distractor dataset: {X.shape[0]:,} samples, label=1 rate={y.mean():.2f}")
    return X, y


def train_distractor_ranker(train_df, val_df, vectorizer):
    """Train LR + RF distractor ranker; keep best model."""
    print("[3/7] Training Distractor Ranker …")

    X_train, y_train = build_distractor_dataset(train_df, vectorizer, max_rows=40_000)
    X_val,   y_val   = build_distractor_dataset(val_df,   vectorizer, max_rows=5_000)

    # --- Logistic Regression (with grid search) ---
    print("    GridSearch over LR …")
    param_grid = {"C": [0.1, 1.0, 5.0, 10.0]}
    lr_base = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    gs_lr   = GridSearchCV(lr_base, param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=0)
    gs_lr.fit(X_train, y_train)
    lr_best = gs_lr.best_estimator_
    print(f"    LR best C={gs_lr.best_params_['C']}  CV F1={gs_lr.best_score_:.4f}")

    # --- Random Forest ---
    print("    Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, n_jobs=-1, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    # Evaluate both on validation set
    results = {}
    for name, model in [("LR", lr_best), ("RF", rf)]:
        y_pred = model.predict(X_val)
        results[name] = {
            "accuracy":  accuracy_score(y_val, y_pred),
            "macro_f1":  f1_score(y_val, y_pred, average="macro"),
            "precision": precision_score(y_val, y_pred, average="macro", zero_division=0),
            "recall":    recall_score(y_val, y_pred, average="macro", zero_division=0),
        }
        print(f"\n    [{name}] Val Metrics:")
        for k, v in results[name].items():
            print(f"      {k:12s}: {v:.4f}")
        print("    Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))

    # Pick the better model by macro F1
    best_name  = max(results, key=lambda n: results[n]["macro_f1"])
    best_model = lr_best if best_name == "LR" else rf
    print(f"\n    Best distractor ranker: {best_name}")

    out_path = f"{MODEL_B_DIR}/distractor_ranker.pkl"
    joblib.dump(best_model, out_path)
    print(f"    Saved → {out_path}")

    return best_model, X_val, y_val, results

# ---------------------------------------------------------------------------
# 4. HINT SCORER PIPELINE  (Section 7.6)
# ---------------------------------------------------------------------------

def build_hint_features(sentence, question, article, idx, n_sents, vectorizer):
    """
    Features per sentence (Section 7.6 ML-Scored Hint Strategy):
      - keyword overlap count (question tokens ∩ sentence tokens)
      - sentence position in article (normalised 0–1)
      - sentence length (word count)
      - cosine similarity to question (One-Hot / TF-IDF based)
    """
    q_toks   = tokenize(question)
    s_toks   = tokenize(sentence)
    overlap  = keyword_overlap(q_toks, s_toks)
    position = idx / max(n_sents - 1, 1)
    length   = len(s_toks)

    try:
        q_vec  = vectorizer.transform([clean_text(question)])
        s_vec  = vectorizer.transform([clean_text(sentence)])
        sim    = float(cosine_similarity(q_vec, s_vec)[0, 0])
    except Exception:
        sim = 0.0

    # Jaccard between question tokens and sentence tokens
    jac = jaccard(q_toks, s_toks)

    # Named-entity proxy: count capitalised words in the original sentence
    cap_words = len(re.findall(r"\b[A-Z][a-z]+\b", sentence))

    return np.array([overlap, position, length, sim, jac, cap_words], dtype=np.float32)


def build_hint_dataset(df, vectorizer, max_rows=20_000):
    """
    For every (article, question, correct_answer) we label each sentence of
    the article:
      label = 1 if the sentence contains the correct answer text, else 0
    """
    print("    Building hint feature matrix …")
    X_rows, y_rows, sim_rows = [], [], []

    for i, row in df.iterrows():
        if i >= max_rows:
            break
        article     = row["article"]
        question    = row["question"]
        correct_ans = clean_text(row[row["answer"]])
        sentences   = split_sentences(article)
        n           = len(sentences)

        if n < 2:
            continue

        for idx, sent in enumerate(sentences):
            # Label: does this sentence contain the correct answer?
            label = int(correct_ans in clean_text(sent))
            feats = build_hint_features(sent, question, article, idx, n, vectorizer)
            X_rows.append(feats)
            y_rows.append(label)
            # Store cosine sim for R² evaluation
            sim_rows.append(feats[3])   # index 3 is cosine_sim

    X = np.vstack(X_rows)
    y = np.array(y_rows)
    print(f"    Hint dataset: {X.shape[0]:,} sentences, label=1 rate={y.mean():.3f}")
    return X, y, np.array(sim_rows)


def train_hint_scorer(train_df, val_df, vectorizer):
    """Train a Logistic Regression hint sentence scorer."""
    print("[4/7] Training Hint Scorer …")

    X_train, y_train, sim_train = build_hint_dataset(train_df, vectorizer, max_rows=30_000)
    X_val,   y_val,   sim_val   = build_hint_dataset(val_df,   vectorizer, max_rows=5_000)

    param_grid = {"C": [0.01, 0.1, 1.0, 5.0]}
    lr_base = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    gs      = GridSearchCV(lr_base, param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    scorer = gs.best_estimator_
    print(f"    HintScorer best C={gs.best_params_['C']}  CV F1={gs.best_score_:.4f}")

    y_pred     = scorer.predict(X_val)
    y_prob     = scorer.predict_proba(X_val)[:, 1]

    print(f"\n    [HintScorer] Val Metrics:")
    print(f"      accuracy : {accuracy_score(y_val, y_pred):.4f}")
    print(f"      macro_f1 : {f1_score(y_val, y_pred, average='macro'):.4f}")
    print(f"      precision: {precision_score(y_val, y_pred, average='macro', zero_division=0):.4f}")
    print(f"      recall   : {recall_score(y_val, y_pred, average='macro', zero_division=0):.4f}")

    # R² — correlation of predicted probabilities vs true cosine similarity
    r2 = r2_score(sim_val, y_prob)
    print(f"      R²       : {r2:.4f}  (predicted prob vs cosine sim to question)")

    # Precision@K for hint retrieval
    for k in [1, 3, 5]:
        pk = precision_at_k(y_val, y_prob, k=k)
        print(f"      P@{k}      : {pk:.4f}")

    print("    Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    out_path = f"{MODEL_B_DIR}/hint_scorer.pkl"
    joblib.dump(scorer, out_path)
    print(f"    Saved → {out_path}")
    return scorer


def precision_at_k(y_true, y_scores, k=3):
    """Precision@K: fraction of top-K ranked sentences that truly contain answer."""
    if len(y_true) < k:
        return 0.0
    top_k_idx = np.argsort(y_scores)[-k:]
    return y_true[top_k_idx].mean()

# ---------------------------------------------------------------------------
# 5. VECTORIZER (load from Model A or build fresh)
# ---------------------------------------------------------------------------

def load_or_build_vectorizer(train_df):
    """
    Try to load the TF-IDF vectorizer saved by model_a_train.py.
    If it doesn't exist, build a fresh one on the RACE training corpus.
    """
    tfidf_path = f"{MODEL_A_DIR}/tfidf_vectorizer.pkl"
    if os.path.exists(tfidf_path):
        print(f"    Loading TF-IDF vectorizer from {tfidf_path}")
        return joblib.load(tfidf_path)

    print("    TF-IDF vectorizer not found — building fresh …")
    corpus = []
    for _, row in train_df.iterrows():
        combined = " ".join([row["article"], row["question"],
                             row["A"], row["B"], row["C"], row["D"]])
        corpus.append(clean_text(combined))

    vectorizer = TfidfVectorizer(
        max_features=10_000,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        norm="l2",
    )
    vectorizer.fit(corpus)
    os.makedirs(MODEL_A_DIR, exist_ok=True)
    joblib.dump(vectorizer, tfidf_path)
    print(f"    Saved fresh TF-IDF vectorizer → {tfidf_path}")
    return vectorizer

# ---------------------------------------------------------------------------
# 6. FULL EVALUATION ON TEST SET  (Section 7.8)
# ---------------------------------------------------------------------------

def evaluate_on_test(test_df, distractor_ranker, hint_scorer, vectorizer):
    """
    Run both models on the held-out test set and print all required metrics.
    """
    print("[6/7] Full Evaluation on Test Set …")

    # -- Distractor Ranker --
    X_test_d, y_test_d = build_distractor_dataset(test_df, vectorizer, max_rows=10_000)
    y_pred_d = distractor_ranker.predict(X_test_d)

    print("\n  === Distractor Ranker — Test Set ===")
    print(f"  Accuracy : {accuracy_score(y_test_d, y_pred_d):.4f}")
    print(f"  Macro F1 : {f1_score(y_test_d, y_pred_d, average='macro'):.4f}")
    print(f"  Precision: {precision_score(y_test_d, y_pred_d, average='macro', zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_test_d, y_pred_d, average='macro', zero_division=0):.4f}")
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test_d, y_pred_d))

    # -- Hint Scorer --
    X_test_h, y_test_h, sim_test = build_hint_dataset(test_df, vectorizer, max_rows=5_000)
    y_pred_h = hint_scorer.predict(X_test_h)
    y_prob_h = hint_scorer.predict_proba(X_test_h)[:, 1]

    print("\n  === Hint Scorer — Test Set ===")
    print(f"  Accuracy : {accuracy_score(y_test_h, y_pred_h):.4f}")
    print(f"  Macro F1 : {f1_score(y_test_h, y_pred_h, average='macro'):.4f}")
    print(f"  Precision: {precision_score(y_test_h, y_pred_h, average='macro', zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_test_h, y_pred_h, average='macro', zero_division=0):.4f}")
    print(f"  R²       : {r2_score(sim_test, y_prob_h):.4f}")
    for k in [1, 3, 5]:
        print(f"  P@{k}      : {precision_at_k(y_test_h, y_prob_h, k=k):.4f}")
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test_h, y_pred_h))

# ---------------------------------------------------------------------------
# 7. INFERENCE API — used by ui/app.py  (Section 12)
# ---------------------------------------------------------------------------

class ModelBPipeline:
    """
    Clean inference API. Loads saved artefacts and exposes:
      - generate_distractors(article, question, correct_answer)
      - generate_hints(article, question, correct_answer)
    Both follow the exact signatures from Section 12 of the spec.
    """

    def __init__(self):
        self.vectorizer       = joblib.load(f"{MODEL_A_DIR}/tfidf_vectorizer.pkl")
        self.distractor_ranker = joblib.load(f"{MODEL_B_DIR}/distractor_ranker.pkl")
        self.hint_scorer       = joblib.load(f"{MODEL_B_DIR}/hint_scorer.pkl")

        w2v_path = f"{MODEL_B_DIR}/word2vec_model.bin"
        if os.path.exists(w2v_path):
            from gensim.models import Word2Vec as _W2V
            self.w2v = _W2V.load(w2v_path).wv
        else:
            self.w2v = None

    # ------------------------------------------------------------------
    # Distractor generation (Section 7.3 + 7.5)
    # ------------------------------------------------------------------

    def generate_distractors(self, article: str, question: str,
                              correct_answer: str, num_distractors: int = 3):
        """
        Step 1: Extract candidates from passage (frequency-based).
        Step 2: Score each candidate with the trained ranker.
        Step 3: Return top-N that are not the correct answer.
        Fallback: Word2Vec nearest neighbours if < 3 candidates found.
        """
        candidates = self._extract_candidates(article, correct_answer)
        if not candidates:
            return self._fallback_distractors(correct_answer, article, num_distractors)

        correct_clean = clean_text(correct_answer)
        scored = []
        for cand in candidates:
            row_dict = {
                "article":  article,
                "question": question,
                "candidate": cand,
                "answer":   "A",          # dummy key — correct_ans passed directly
                "A": correct_answer, "B": "", "C": "", "D": "",
            }
            feats = build_distractor_features(row_dict, self.vectorizer,
                                              answer_col=correct_clean)
            score = self.distractor_ranker.predict_proba(feats.reshape(1, -1))[0, 1]
            scored.append((score, cand))

        scored.sort(reverse=True)

        # Diversity filter (spec §7.4): reject if cosine_sim > 0.7 with already chosen
        chosen = []
        for score, cand in scored:
            if len(chosen) == num_distractors:
                break
            if not self._too_similar(cand, chosen):
                chosen.append(cand)

        # Pad with Word2Vec fallbacks
        if len(chosen) < num_distractors:
            chosen += self._fallback_distractors(
                correct_answer, article, num_distractors - len(chosen)
            )

        return chosen[:num_distractors]

    def _extract_candidates(self, passage: str, answer: str):
        """Section 7.3 Step 1 — frequency-based candidate extraction."""
        tokens = tokenize(passage)
        freq   = Counter(tokens)
        ans_tokens = set(tokenize(answer))
        candidates = [
            w for w, _ in freq.most_common(80)
            if w not in ans_tokens and w not in STOPWORDS and len(w) > 3
        ]
        # Also add short noun-phrases (capitalised bigrams from original passage)
        caps = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", passage)
        candidates += [c.lower() for c in caps if c.lower() not in answer.lower()]
        return list(dict.fromkeys(candidates))  # deduplicate while preserving order

    def _too_similar(self, candidate: str, chosen: list, threshold: float = 0.7):
        """Diversity penalty from spec §7.4."""
        if not chosen:
            return False
        try:
            cand_vec   = self.vectorizer.transform([clean_text(candidate)])
            chosen_vec = self.vectorizer.transform([clean_text(c) for c in chosen])
            sims = cosine_similarity(cand_vec, chosen_vec).flatten()
            return bool((sims > threshold).any())
        except Exception:
            return False

    def _fallback_distractors(self, answer: str, passage: str, n: int):
        """Word2Vec nearest-neighbour fallback (spec §7.5)."""
        if self.w2v is None:
            return ["Information not provided in text."] * n

        words   = tokenize(answer)
        results = []
        for w in words:
            if w in self.w2v:
                neighbours = self.w2v.most_similar(w, topn=20)
                for nb_word, _ in neighbours:
                    if nb_word.lower() not in passage.lower() and nb_word not in results:
                        results.append(nb_word)
                    if len(results) == n:
                        break
            if len(results) == n:
                break

        while len(results) < n:
            results.append("Information not provided in text.")
        return results[:n]

    # ------------------------------------------------------------------
    # Hint generation (Section 7.6)
    # ------------------------------------------------------------------

    def generate_hints(self, article: str, question: str,
                       correct_answer: str, n_hints: int = 3):
        """
        Score every sentence with the ML hint scorer.
        Return hints ordered from MOST GENERAL (lowest score) to
        NEAR-EXPLICIT (highest score), matching spec §7.6 graduation rules.
        """
        sentences = split_sentences(article)
        if len(sentences) <= 1:
            return sentences * n_hints

        n = len(sentences)
        feats_list = [
            build_hint_features(s, question, article, i, n, self.vectorizer)
            for i, s in enumerate(sentences)
        ]
        X    = np.vstack(feats_list)
        probs = self.hint_scorer.predict_proba(X)[:, 1]

        # Pick top-3 unique indices sorted by score ascending
        ranked_idx = np.argsort(probs)[::-1]
        top_idx    = []
        for idx in ranked_idx:
            top_idx.append(int(idx))
            if len(top_idx) == n_hints:
                break

        # Sort ascending so Hint 1 is least relevant (most general)
        top_idx.sort(key=lambda i: probs[i])

        hints = [sentences[i] for i in top_idx]
        return hints   # index 0 = broadest, index 2 = near-explicit

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 65)
    print("  MODEL B — FULL TRAINING PIPELINE")
    print("=" * 65)

    # 1. Load data
    train_df, val_df, test_df = load_data()

    # 2. Vectorizer
    print("[2/7] Loading / building vectorizer …")
    vectorizer = load_or_build_vectorizer(train_df)

    # 3. Word2Vec
    w2v_model = train_word2vec(train_df, val_df)

    # 4. Distractor Ranker
    distractor_ranker, X_val_d, y_val_d, dist_results = \
        train_distractor_ranker(train_df, val_df, vectorizer)

    # 5. Hint Scorer
    hint_scorer = train_hint_scorer(train_df, val_df, vectorizer)

    # 6. Test evaluation
    evaluate_on_test(test_df, distractor_ranker, hint_scorer, vectorizer)

    # 7. Smoke-test the inference API
    print("[7/7] Smoke-testing inference API …")
    sample = val_df.iloc[0]
    pipeline = ModelBPipeline()

    t_inf = time.time()
    distractors = pipeline.generate_distractors(
        article=sample["article"],
        question=sample["question"],
        correct_answer=sample[sample["answer"]],
    )
    hints = pipeline.generate_hints(
        article=sample["article"],
        question=sample["question"],
        correct_answer=sample[sample["answer"]],
    )
    latency_ms = (time.time() - t_inf) * 1000

    print(f"\n  Question : {sample['question']}")
    print(f"  Correct  : {sample[sample['answer']]}")
    print("\n  Generated Distractors:")
    for i, d in enumerate(distractors, 1):
        print(f"    ❌ Option {i}: {d}")
    print("\n  Generated Hints (Hint 1 = most general):")
    for i, h in enumerate(hints, 1):
        print(f"    💡 Hint {i}: {h[:120]}…" if len(h) > 120 else f"    💡 Hint {i}: {h}")

    print(f"\n  Inference latency: {latency_ms:.1f} ms")
    print(f"\nTotal training time: {(time.time() - t0) / 60:.1f} min")
    print("=" * 65)
    print("  Model B training COMPLETE.")
    print(f"  Artefacts saved in: {MODEL_B_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()