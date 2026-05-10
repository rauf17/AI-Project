# =============================================================================
# src/inference.py — Unified Inference API
# Used exclusively by ui/app.py and tests/test_inference.py
#
# Exposes three public functions (spec §12):
#   predict_answer(article, question, options)  → dict
#   generate_distractors(article, question, correct_answer)  → list[str]
#   generate_hints(article, question, correct_answer)  → list[str]
#
# Also exposes:
#   load_pipeline()  → InferencePipeline (singleton, cached)
#   random_race_sample(split)  → dict
# =============================================================================

from __future__ import annotations

import os
import re
import time
import warnings
import random
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths — adjust if your project root differs
# ---------------------------------------------------------------------------
_ROOT       = os.path.dirname(os.path.abspath(__file__))
_PROJ       = os.path.dirname(_ROOT) if os.path.basename(_ROOT) == "src" else _ROOT
_MODEL_A    = os.path.join(_PROJ, "models", "model_a", "traditional")
_MODEL_B    = os.path.join(_PROJ, "models", "model_b", "traditional")
_DATA_RAW   = os.path.join(_PROJ, "data", "raw")

# ---------------------------------------------------------------------------
# Text helpers (duplicated from preprocessing to keep inference self-contained)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "i","me","my","we","our","you","your","he","him","his","she","her",
    "it","its","they","them","their","what","which","who","this","that",
    "these","those","am","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","a","an","the","and","but",
    "if","or","as","of","at","by","for","with","in","out","on","to",
    "from","up","down","not","no","so","than","very","just","will",
    "can","don","should","now","s","t","d","ll","m","o","re","ve","y",
}

def _clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _tokenize(text: str) -> List[str]:
    return [w for w in _clean(text).split() if w not in _STOPWORDS and len(w) > 1]

def _split_sentences(article: str) -> List[str]:
    raw = re.split(r"(?<=[.!?])\s+", article.replace("\n", " "))
    return [s.strip() for s in raw if len(s.strip()) > 10]

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)

# ---------------------------------------------------------------------------
# InferencePipeline — loads all artefacts once, reuses them
# ---------------------------------------------------------------------------

class InferencePipeline:
    """
    Holds all loaded model artefacts.
    Instantiate once (via load_pipeline()) and reuse across requests.
    """

    def __init__(self):
        self._ready = False
        self._error: Optional[str] = None

        # Model A artefacts
        self.tfidf_vectorizer   = None
        self.onehot_encoder     = None
        self.lr_classifier      = None
        self.svm_classifier     = None
        self.nb_classifier      = None
        self.rf_classifier      = None
        self.xgb_classifier     = None
        self.ensemble_model     = None   # best ensemble available

        # ── NEW: number of extra hand-crafted features appended at training.
        # Loaded from a small metadata file saved alongside the model.
        # Falls back to auto-detection so old checkpoints still work.
        self._n_extra_features: Optional[int] = None

        # Model B artefacts
        self.distractor_ranker  = None
        self.hint_scorer        = None
        self.w2v                = None

        self._load_all()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _try_load(self, path: str):
        """Load a joblib pickle; return None (not raise) if absent."""
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as exc:
                print(f"[inference] Warning: could not load {path}: {exc}")
        return None

    def _load_all(self):
        # Vectorizers
        self.tfidf_vectorizer  = self._try_load(f"{_MODEL_A}/tfidf_vectorizer.pkl")
        self.onehot_encoder    = self._try_load(f"{_MODEL_A}/onehot_encoder.pkl")

        # Prefer TF-IDF for everything; fall back to one-hot
        self.vectorizer = self.tfidf_vectorizer or self.onehot_encoder

        # Model A classifiers
        self.lr_classifier     = self._try_load(f"{_MODEL_A}/lr_classifier.pkl")
        self.svm_classifier    = self._try_load(f"{_MODEL_A}/svm_classifier.pkl")
        self.nb_classifier     = self._try_load(f"{_MODEL_A}/nb_classifier.pkl")
        self.rf_classifier     = self._try_load(f"{_MODEL_A}/rf_classifier.pkl")
        self.xgb_classifier    = self._try_load(f"{_MODEL_A}/xgb_classifier.pkl")

        # Best ensemble — try meta stacking first, then soft/hard vote
        self.ensemble_model = (
            self._try_load(f"{_MODEL_A}/ensemble_meta.pkl")
            or self._try_load(f"{_MODEL_A}/ensemble_soft.pkl")
            or self._try_load(f"{_MODEL_A}/ensemble_hard.pkl")
        )

        # Primary verifier: ensemble > LR > SVM (first available)
        self.primary_verifier = (
            self.ensemble_model
            or self.lr_classifier
            or self.svm_classifier
            or self.rf_classifier
        )

        # ── Load feature metadata so we know how many extra cols were appended
        #    at training time.  Save this file from your training script with:
        #      joblib.dump({"n_extra": N}, f"{_MODEL_A}/feature_meta.pkl")
        meta = self._try_load(f"{_MODEL_A}/feature_meta.pkl")
        if meta and "n_extra" in meta:
            self._n_extra_features = int(meta["n_extra"])

        # If metadata absent, infer from the saved model's expected feature count
        # vs the vectorizer's output dimension.
        if self._n_extra_features is None and self.primary_verifier is not None and self.vectorizer is not None:
            try:
                n_model  = self.primary_verifier.n_features_in_
                n_tfidf  = self.vectorizer.transform([""]).shape[1]
                n_extra  = n_model - n_tfidf
                if 0 <= n_extra <= 20:          # sanity check
                    self._n_extra_features = n_extra
                    print(f"[inference] Auto-detected n_extra_features = {n_extra}")
                else:
                    # Cannot reconcile — fall back to plain TF-IDF only (0 extras)
                    self._n_extra_features = 0
                    print(f"[inference] Warning: unexpected feature delta {n_extra}; "
                          f"using 0 extra features. Consider retraining.")
            except Exception as exc:
                self._n_extra_features = 0
                print(f"[inference] Could not auto-detect extra features: {exc}")

        # Default safe value
        if self._n_extra_features is None:
            self._n_extra_features = 0

        # Model B
        self.distractor_ranker = self._try_load(f"{_MODEL_B}/distractor_ranker.pkl")
        self.hint_scorer       = self._try_load(f"{_MODEL_B}/hint_scorer.pkl")

        w2v_path = f"{_MODEL_B}/word2vec_model.bin"
        if os.path.exists(w2v_path):
            try:
                from gensim.models import Word2Vec as _W2V
                self.w2v = _W2V.load(w2v_path).wv
            except Exception:
                self.w2v = None

        if self.vectorizer is None:
            self._error = (
                "No vectorizer found. Run `python src/preprocessing.py` "
                "and `python src/model_a_train.py` first."
            )
        elif self.primary_verifier is None:
            self._error = (
                "No trained classifier found. "
                "Run `python src/model_a_train.py` first."
            )
        else:
            self._ready = True

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def error_message(self) -> Optional[str]:
        return self._error

    # ------------------------------------------------------------------
    # Feature builders (self-contained, no external imports)
    # ------------------------------------------------------------------

    def _make_verification_features(self, article: str, question: str,
                                    option: str) -> sp.csr_matrix:
        """
        Build the same combined feature vector used during Model A training.

        The number of hand-crafted extra columns appended after the TF-IDF
        vector is determined by self._n_extra_features (auto-detected from
        the saved model at load time), so this always matches what the
        classifier was trained on — regardless of how many extras were used.
        """
        combined = f"{article} {article} {question} {option}"
        X = self.vectorizer.transform([_clean(combined)])

        n_extra = self._n_extra_features

        if n_extra == 0:
            # Model was trained on raw TF-IDF only — return as-is
            return X

        # Pre-compute vectors needed for extra features
        art_vec  = self.vectorizer.transform([_clean(article)])
        q_vec    = self.vectorizer.transform([_clean(question)])
        opt_vec  = self.vectorizer.transform([_clean(option)])

        cos_art_opt = cosine_similarity(art_vec, opt_vec)[0, 0]
        cos_q_opt   = cosine_similarity(q_vec,   opt_vec)[0, 0]
        cos_art_q   = cosine_similarity(art_vec, q_vec  )[0, 0]

        q_toks = _tokenize(question)
        o_toks = _tokenize(option)
        a_toks = _tokenize(article)

        # Full pool of 6 hand-crafted features (same order as training script)
        all_extra = [
            cos_art_opt,
            cos_q_opt,
            cos_art_q,
            _jaccard(q_toks, o_toks),
            len(o_toks) / max(len(a_toks), 1),
            len(set(q_toks) & set(o_toks)),
        ]

        # Slice to exactly n_extra columns so we always match training
        extra = np.array([all_extra[:n_extra]], dtype=np.float32)
        return sp.hstack([X, sp.csr_matrix(extra)], format="csr")

    def _make_distractor_features(self, article: str, question: str,
                                  candidate: str, correct_answer: str) -> np.ndarray:
        correct_clean = _clean(correct_answer)
        cand_clean    = _clean(candidate)

        try:
            ans_vec  = self.vectorizer.transform([correct_clean])
            cand_vec = self.vectorizer.transform([cand_clean])
            art_vec  = self.vectorizer.transform([_clean(article)])
            sim_ans_cand = float(cosine_similarity(ans_vec,  cand_vec)[0, 0])
            sim_art_cand = float(cosine_similarity(art_vec,  cand_vec)[0, 0])
        except Exception:
            sim_ans_cand = sim_art_cand = 0.0

        chars_a   = set(correct_clean)
        chars_c   = set(cand_clean)
        char_ov   = len(chars_a & chars_c) / max(len(chars_a | chars_c), 1)

        art_tok   = _tokenize(article)
        cand_tok  = _tokenize(candidate)
        art_freq  = Counter(art_tok)
        cand_freq = sum(art_freq.get(t, 0) for t in cand_tok) / max(len(cand_tok), 1)

        art_lower = article.lower()
        first_w   = (cand_tok[0] if cand_tok else "")
        pos       = art_lower.find(first_w)
        norm_pos  = (pos / max(len(art_lower), 1)) if pos >= 0 else 1.0

        len_diff  = abs(len(cand_tok) - len(_tokenize(correct_answer))) / max(len(_tokenize(correct_answer)), 1)
        jac       = _jaccard(cand_tok, _tokenize(correct_answer))

        return np.array([sim_ans_cand, sim_art_cand, char_ov,
                         cand_freq, norm_pos, len_diff, jac], dtype=np.float32)

    def _make_hint_features(self, sentence: str, question: str,
                             article: str, idx: int, n_sents: int) -> np.ndarray:
        q_toks  = _tokenize(question)
        s_toks  = _tokenize(sentence)
        overlap = len(set(q_toks) & set(s_toks))
        pos     = idx / max(n_sents - 1, 1)
        length  = len(s_toks)
        jac     = _jaccard(q_toks, s_toks)
        caps    = len(re.findall(r"\b[A-Z][a-z]+\b", sentence))

        try:
            q_vec = self.vectorizer.transform([_clean(question)])
            s_vec = self.vectorizer.transform([_clean(sentence)])
            sim   = float(cosine_similarity(q_vec, s_vec)[0, 0])
        except Exception:
            sim = 0.0

        return np.array([overlap, pos, length, sim, jac, caps], dtype=np.float32)

    # ------------------------------------------------------------------
    # Public inference methods
    # ------------------------------------------------------------------

    def verify_option(self, article: str, question: str,
                      option: str) -> Dict[str, float]:
        """
        Returns {"probability_correct": float, "prediction": int (0 or 1)}
        """
        if not self._ready:
            raise RuntimeError(self._error)

        X = self._make_verification_features(article, question, option)

        try:
            prob = float(self.primary_verifier.predict_proba(X)[0, 1])
        except AttributeError:
            # Models without predict_proba (e.g. hard LinearSVC)
            pred = int(self.primary_verifier.predict(X)[0])
            prob = float(pred)

        return {"probability_correct": prob, "prediction": int(prob >= 0.5)}

    def rank_options(self, article: str, question: str,
                     options: Dict[str, str]) -> Dict[str, float]:
        scores = {}
        for key, text in options.items():
            scores[key] = self.verify_option(article, question, text)["probability_correct"]
        return scores

    def generate_distractors(self, article: str, question: str,
                              correct_answer: str,
                              num_distractors: int = 3) -> List[str]:
        if not self._ready:
            raise RuntimeError(self._error)

        if self.distractor_ranker is None:
            return self._cosine_distractors(article, correct_answer, num_distractors)

        candidates = self._extract_candidates(article, correct_answer)
        if not candidates:
            return self._w2v_distractors(correct_answer, article, num_distractors)

        scored = []
        for cand in candidates:
            feats = self._make_distractor_features(article, question, cand, correct_answer)
            try:
                score = float(self.distractor_ranker.predict_proba(feats.reshape(1, -1))[0, 1])
            except AttributeError:
                score = float(self.distractor_ranker.predict(feats.reshape(1, -1))[0])
            scored.append((score, cand))

        scored.sort(reverse=True)

        chosen: List[str] = []
        for _, cand in scored:
            if len(chosen) == num_distractors:
                break
            if not self._too_similar(cand, chosen):
                chosen.append(cand)

        if len(chosen) < num_distractors:
            chosen += self._w2v_distractors(
                correct_answer, article, num_distractors - len(chosen)
            )

        return chosen[:num_distractors]

    def generate_hints(self, article: str, question: str,
                       correct_answer: str, n_hints: int = 3) -> List[str]:
        if not self._ready:
            raise RuntimeError(self._error)

        sentences = _split_sentences(article)
        if len(sentences) < 2:
            return (sentences * n_hints)[:n_hints]

        n = len(sentences)

        if self.hint_scorer is not None:
            feats_list = [
                self._make_hint_features(s, question, article, i, n)
                for i, s in enumerate(sentences)
            ]
            X     = np.vstack(feats_list)
            probs = self.hint_scorer.predict_proba(X)[:, 1]
        else:
            probs = self._cosine_sentence_scores(sentences, question)

        ranked_idx = np.argsort(probs)[::-1]
        top_idx: List[int] = []
        for idx in ranked_idx:
            top_idx.append(int(idx))
            if len(top_idx) == n_hints:
                break

        top_idx.sort(key=lambda i: probs[i])
        return [sentences[i] for i in top_idx]

    def supporting_sentence(self, article: str, correct_answer: str) -> str:
        sentences = _split_sentences(article)
        if not sentences:
            return ""

        ans_clean = _clean(correct_answer)
        for s in sentences:
            if ans_clean in _clean(s):
                return s

        scores = self._cosine_sentence_scores(sentences, correct_answer)
        return sentences[int(np.argmax(scores))]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_candidates(self, passage: str, answer: str) -> List[str]:
        tokens    = _tokenize(passage)
        freq      = Counter(tokens)
        ans_toks  = set(_tokenize(answer))
        candidates = [
            w for w, _ in freq.most_common(100)
            if w not in ans_toks and len(w) > 3
        ]
        caps = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", passage)
        candidates += [c.lower() for c in caps if c.lower() not in answer.lower()]
        return list(dict.fromkeys(candidates))

    def _too_similar(self, candidate: str, chosen: List[str],
                     threshold: float = 0.7) -> bool:
        if not chosen:
            return False
        try:
            cand_vec   = self.vectorizer.transform([_clean(candidate)])
            chosen_vec = self.vectorizer.transform([_clean(c) for c in chosen])
            sims = cosine_similarity(cand_vec, chosen_vec).flatten()
            return bool((sims > threshold).any())
        except Exception:
            return False

    def _cosine_distractors(self, article: str, correct_answer: str,
                             n: int) -> List[str]:
        sentences = _split_sentences(article)
        if not sentences:
            return ["Information not available."] * n

        try:
            ans_vec = self.vectorizer.transform([_clean(correct_answer)])
            s_vecs  = self.vectorizer.transform([_clean(s) for s in sentences])
            sims    = cosine_similarity(ans_vec, s_vecs).flatten()
        except Exception:
            sims = np.zeros(len(sentences))

        ranked = np.argsort(sims)
        distractors: List[str] = []
        for idx in ranked:
            cand = sentences[idx]
            if (_clean(correct_answer) not in _clean(cand) and
                    not self._too_similar(cand, distractors)):
                distractors.append(cand[:120])
            if len(distractors) == n:
                break

        while len(distractors) < n:
            distractors.append("Information not provided in text.")
        return distractors

    def _w2v_distractors(self, answer: str, passage: str, n: int) -> List[str]:
        if self.w2v is None:
            return ["Information not provided in text."] * n

        results: List[str] = []
        for w in _tokenize(answer):
            if w in self.w2v:
                for nb_word, _ in self.w2v.most_similar(w, topn=30):
                    if (nb_word.lower() not in passage.lower()
                            and nb_word not in results):
                        results.append(nb_word)
                    if len(results) == n:
                        break
            if len(results) == n:
                break

        while len(results) < n:
            results.append("Information not provided in text.")
        return results[:n]

    def _cosine_sentence_scores(self, sentences: List[str],
                                 query: str) -> np.ndarray:
        try:
            q_vec  = self.vectorizer.transform([_clean(query)])
            s_vecs = self.vectorizer.transform([_clean(s) for s in sentences])
            return cosine_similarity(q_vec, s_vecs).flatten()
        except Exception:
            return np.zeros(len(sentences))


# ---------------------------------------------------------------------------
# Singleton cache
# ---------------------------------------------------------------------------

_pipeline_instance: Optional[InferencePipeline] = None

def load_pipeline() -> InferencePipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = InferencePipeline()
    return _pipeline_instance


# ---------------------------------------------------------------------------
# Public functional API — spec §12
# ---------------------------------------------------------------------------

def predict_answer(article: str, question: str,
                   options: Dict[str, str]) -> Dict:
    pipeline = load_pipeline()
    if not pipeline.is_ready:
        return {"error": pipeline.error_message}

    t0     = time.perf_counter()
    scores = pipeline.rank_options(article, question, options)
    predicted = max(scores, key=scores.get)
    latency   = (time.perf_counter() - t0) * 1000

    return {
        "predicted":   predicted,
        "confidence":  round(scores[predicted], 4),
        "scores":      {k: round(v, 4) for k, v in scores.items()},
        "latency_ms":  round(latency, 1),
    }


def generate_distractors(article: str, question: str,
                         correct_answer: str) -> List[str]:
    pipeline = load_pipeline()
    if not pipeline.is_ready:
        raise RuntimeError(pipeline.error_message)
    return pipeline.generate_distractors(article, question, correct_answer)


def generate_hints(article: str, question: str,
                   correct_answer: str) -> List[str]:
    pipeline = load_pipeline()
    if not pipeline.is_ready:
        raise RuntimeError(pipeline.error_message)
    return pipeline.generate_hints(article, question, correct_answer)


# ---------------------------------------------------------------------------
# RACE sample loader
# ---------------------------------------------------------------------------

def random_race_sample(split: str = "val") -> Dict:
    path = os.path.join(_DATA_RAW, f"{split}.csv")
    if not os.path.exists(path):
        path = os.path.join(_DATA_RAW, "train.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"RACE CSV not found at {path}. "
            "Download from Kaggle and place in data/raw/."
        )

    df  = pd.read_csv(path)
    row = df.sample(1).iloc[0]
    return {
        "article":  str(row["article"]),
        "question": str(row["question"]),
        "options":  {
            "A": str(row["A"]),
            "B": str(row["B"]),
            "C": str(row["C"]),
            "D": str(row["D"]),
        },
        "answer": str(row["answer"]),
    }


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading pipeline…")
    p = load_pipeline()
    print(f"Ready       : {p.is_ready}")
    print(f"n_extra_feats: {p._n_extra_features}")
    if p.error_message:
        print(f"Error: {p.error_message}")
    else:
        sample = random_race_sample("val")
        print(f"\nQuestion: {sample['question']}")
        result = predict_answer(sample["article"], sample["question"], sample["options"])
        print(f"predict_answer → {result}")