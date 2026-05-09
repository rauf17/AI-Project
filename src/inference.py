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
        Build the same combined feature vector used during Model A training:
          article + article + question + option  (article weighted 2×)
        """
        combined = f"{article} {article} {question} {option}"
        X = self.vectorizer.transform([_clean(combined)])

        # Append cosine similarity features as extra columns
        art_vec  = self.vectorizer.transform([_clean(article)])
        q_vec    = self.vectorizer.transform([_clean(question)])
        opt_vec  = self.vectorizer.transform([_clean(option)])

        cos_art_opt = cosine_similarity(art_vec, opt_vec)[0, 0]
        cos_q_opt   = cosine_similarity(q_vec,   opt_vec)[0, 0]
        cos_art_q   = cosine_similarity(art_vec, q_vec  )[0, 0]

        q_toks = _tokenize(question)
        o_toks = _tokenize(option)
        a_toks = _tokenize(article)

        extra = np.array([[
            cos_art_opt,
            cos_q_opt,
            cos_art_q,
            _jaccard(q_toks, o_toks),
            len(o_toks) / max(len(a_toks), 1),
            len(set(q_toks) & set(o_toks)),
        ]], dtype=np.float32)

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
        """
        Score all four options and return {key: prob} dict.
        The key with highest prob is the predicted correct answer.
        """
        scores = {}
        for key, text in options.items():
            scores[key] = self.verify_option(article, question, text)["probability_correct"]
        return scores

    def generate_distractors(self, article: str, question: str,
                              correct_answer: str,
                              num_distractors: int = 3) -> List[str]:
        """
        Returns a list of `num_distractors` plausible-but-wrong answer strings.
        """
        if not self._ready:
            raise RuntimeError(self._error)

        # If no distractor ranker trained yet, fall back to cosine-based extraction
        if self.distractor_ranker is None:
            return self._cosine_distractors(article, correct_answer, num_distractors)

        candidates = self._extract_candidates(article, correct_answer)
        if not candidates:
            return self._w2v_distractors(correct_answer, article, num_distractors)

        # Score every candidate
        scored = []
        for cand in candidates:
            feats = self._make_distractor_features(article, question, cand, correct_answer)
            try:
                score = float(self.distractor_ranker.predict_proba(feats.reshape(1, -1))[0, 1])
            except AttributeError:
                score = float(self.distractor_ranker.predict(feats.reshape(1, -1))[0])
            scored.append((score, cand))

        scored.sort(reverse=True)

        # Diversity filter: cosine similarity threshold 0.7 (spec §7.4)
        chosen: List[str] = []
        for _, cand in scored:
            if len(chosen) == num_distractors:
                break
            if not self._too_similar(cand, chosen):
                chosen.append(cand)

        # Pad with Word2Vec fallback
        if len(chosen) < num_distractors:
            chosen += self._w2v_distractors(
                correct_answer, article, num_distractors - len(chosen)
            )

        return chosen[:num_distractors]

    def generate_hints(self, article: str, question: str,
                       correct_answer: str, n_hints: int = 3) -> List[str]:
        """
        Returns `n_hints` hints sorted from most-general (index 0)
        to near-explicit (index n-1).
        """
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
            # Fallback: pure cosine similarity (original version)
            probs = self._cosine_sentence_scores(sentences, question)

        # Pick top-n_hints unique, sorted ascending (broadest first)
        ranked_idx = np.argsort(probs)[::-1]
        top_idx: List[int] = []
        for idx in ranked_idx:
            top_idx.append(int(idx))
            if len(top_idx) == n_hints:
                break

        top_idx.sort(key=lambda i: probs[i])  # ascending → most general first
        return [sentences[i] for i in top_idx]

    def supporting_sentence(self, article: str, correct_answer: str) -> str:
        """
        Return the passage sentence most likely to contain / support the answer.
        Shown in Screen 2 as explanation after the user submits.
        """
        sentences = _split_sentences(article)
        if not sentences:
            return ""

        ans_clean = _clean(correct_answer)
        for s in sentences:
            if ans_clean in _clean(s):
                return s

        # Fall back to highest cosine sim sentence
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
        # Add capitalised bigrams (proper-noun-like phrases)
        caps = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", passage)
        candidates += [c.lower() for c in caps if c.lower() not in answer.lower()]
        return list(dict.fromkeys(candidates))  # deduplicated

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
        """Pure cosine-similarity distractor fallback (spec §7.4)."""
        sentences = _split_sentences(article)
        if not sentences:
            return ["Information not available."] * n

        try:
            ans_vec = self.vectorizer.transform([_clean(correct_answer)])
            s_vecs  = self.vectorizer.transform([_clean(s) for s in sentences])
            sims    = cosine_similarity(ans_vec, s_vecs).flatten()
        except Exception:
            sims = np.zeros(len(sentences))

        ranked = np.argsort(sims)   # ascending: least similar first
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
    """Return (or create) the global InferencePipeline singleton."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = InferencePipeline()
    return _pipeline_instance


# ---------------------------------------------------------------------------
# Public functional API — spec §12
# ---------------------------------------------------------------------------

def predict_answer(article: str, question: str,
                   options: Dict[str, str]) -> Dict:
    """
    Verify which option is correct.

    Returns:
        {
            "predicted": "A",          # key of predicted correct option
            "confidence": 0.84,        # probability of the predicted option
            "scores": {"A":…,"B":…},   # raw score for all options
            "latency_ms": 47,
        }
    """
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
    """
    Generate three plausible distractor options.

    Returns: ["Played football", "Watched TV", "Stayed home"]
    """
    pipeline = load_pipeline()
    if not pipeline.is_ready:
        raise RuntimeError(pipeline.error_message)

    t0 = time.perf_counter()
    result = pipeline.generate_distractors(article, question, correct_answer)
    _ = (time.perf_counter() - t0) * 1000
    return result


def generate_hints(article: str, question: str,
                   correct_answer: str) -> List[str]:
    """
    Generate three graduated hints.

    Returns: [hint1_general, hint2_specific, hint3_near_explicit]
    """
    pipeline = load_pipeline()
    if not pipeline.is_ready:
        raise RuntimeError(pipeline.error_message)

    return pipeline.generate_hints(article, question, correct_answer)


# ---------------------------------------------------------------------------
# RACE sample loader — used by Screen 1's "Load Random Sample" button
# ---------------------------------------------------------------------------

def random_race_sample(split: str = "val") -> Dict:
    """
    Load a random row from the RACE CSV.

    Returns:
        {
            "article": str,
            "question": str,
            "options": {"A": str, "B": str, "C": str, "D": str},
            "answer": str,   # gold label key, e.g. "B"
        }
    """
    path = os.path.join(_DATA_RAW, f"{split}.csv")
    if not os.path.exists(path):
        # try train as last resort
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
    print(f"Ready: {p.is_ready}")
    if p.error_message:
        print(f"Error: {p.error_message}")
    else:
        sample = random_race_sample("val")
        print(f"\nQuestion: {sample['question']}")
        print(f"Options : {sample['options']}")
        print(f"Gold    : {sample['answer']}")

        result = predict_answer(sample["article"], sample["question"], sample["options"])
        print(f"\npredict_answer → {result}")

        hints = generate_hints(sample["article"], sample["question"],
                               sample["options"][sample["answer"]])
        print("\nHints:")
        for i, h in enumerate(hints, 1):
            print(f"  Hint {i}: {h[:100]}…")

        dist = generate_distractors(sample["article"], sample["question"],
                                    sample["options"][sample["answer"]])
        print("\nDistractors:")
        for d in dist:
            print(f"  ❌ {d[:80]}")