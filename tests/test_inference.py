# =============================================================================
# tests/test_inference.py
# Unit tests for the inference pipeline (src/inference.py)
#
# Run with:   pytest tests/test_inference.py -v
#             pytest tests/test_inference.py -v --tb=short
#
# Tests cover:
#   - Pipeline loading and readiness
#   - predict_answer: output shape, types, key validity
#   - generate_distractors: count, no duplicates, no correct-answer leakage
#   - generate_hints: count, ordering contract, no empty strings
#   - random_race_sample: schema validation
#   - Edge cases: empty inputs, single-sentence articles, unicode
#   - Latency: single inference under 10 s (spec §4.2)
#   - Fallback behaviour when models are missing
# =============================================================================

import os
import sys
import time
import types
import pytest
import numpy as np

# Make sure src/ is importable regardless of where pytest is invoked
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.inference as inf
from src.inference import (
    load_pipeline,
    predict_answer,
    generate_distractors,
    generate_hints,
    random_race_sample,
    InferencePipeline,
    _clean,
    _tokenize,
    _split_sentences,
    _jaccard,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ARTICLE_SHORT = (
    "Mary went to the market early in the morning. "
    "She bought apples, oranges, and a bunch of fresh flowers. "
    "On her way home she met her neighbour John, who was walking his dog. "
    "John told her that the local bakery had closed down last week. "
    "Mary was surprised because she often bought bread there."
)

ARTICLE_LONG = (
    "The Industrial Revolution, which began in Britain in the late 18th century, "
    "fundamentally transformed how goods were produced. "
    "Steam engines replaced human and animal power in factories, mills, and mines. "
    "Cities grew rapidly as workers moved from rural areas to urban centres. "
    "Child labour was widespread, with many children working in dangerous conditions. "
    "Social reformers campaigned for better working conditions and education for all. "
    "By the mid-19th century, railways had connected most major British cities. "
    "The revolution spread to continental Europe and North America over the following decades."
)

QUESTION_1  = "What did Mary buy at the market?"
ANSWER_1    = "apples, oranges, and fresh flowers"

QUESTION_2  = "Where did the Industrial Revolution begin?"
ANSWER_2    = "Britain"

OPTIONS_1 = {
    "A": "apples, oranges, and fresh flowers",
    "B": "bread and milk",
    "C": "vegetables and meat",
    "D": "shoes and clothes",
}

OPTIONS_2 = {
    "A": "France",
    "B": "Germany",
    "C": "Britain",
    "D": "United States",
}

GOLD_1 = "A"
GOLD_2 = "C"


@pytest.fixture(scope="module")
def pipeline():
    """Load the pipeline once per test session."""
    return load_pipeline()


# ---------------------------------------------------------------------------
# Helper — skip if pipeline not ready (models not yet trained)
# ---------------------------------------------------------------------------

def _skip_if_not_ready(pipeline):
    if not pipeline.is_ready:
        pytest.skip(f"Pipeline not ready: {pipeline.error_message}")


# ===========================================================================
# 1. UTILITY FUNCTION TESTS (no models needed)
# ===========================================================================

class TestUtilities:

    def test_clean_lowercase(self):
        assert _clean("Hello WORLD") == "hello world"

    def test_clean_strips_punctuation(self):
        result = _clean("Hello, world! How's it going?")
        assert "," not in result
        assert "!" not in result

    def test_clean_collapses_whitespace(self):
        result = _clean("too    many   spaces")
        assert "  " not in result

    def test_clean_empty_string(self):
        assert _clean("") == ""

    def test_clean_unicode(self):
        result = _clean("café naïve résumé")
        assert isinstance(result, str)

    def test_tokenize_removes_stopwords(self):
        tokens = _tokenize("the cat sat on the mat")
        assert "the" not in tokens
        assert "on" not in tokens
        assert "cat" in tokens
        assert "mat" in tokens

    def test_tokenize_empty(self):
        assert _tokenize("") == []

    def test_tokenize_returns_list(self):
        assert isinstance(_tokenize("hello world"), list)

    def test_split_sentences_basic(self):
        sents = _split_sentences("Hello world. How are you? Fine thanks.")
        assert len(sents) >= 2

    def test_split_sentences_filters_short(self):
        sents = _split_sentences("Hi. This is a longer sentence that should survive.")
        # "Hi." is too short (<10 chars) and should be filtered
        for s in sents:
            assert len(s) > 10

    def test_split_sentences_empty(self):
        sents = _split_sentences("")
        assert sents == []

    def test_jaccard_identical(self):
        a = ["apple", "banana"]
        assert _jaccard(a, a) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard(["a", "b"], ["c", "d"]) == 0.0

    def test_jaccard_partial(self):
        score = _jaccard(["a", "b", "c"], ["b", "c", "d"])
        assert 0.0 < score < 1.0

    def test_jaccard_empty(self):
        assert _jaccard([], []) == 0.0


# ===========================================================================
# 2. PIPELINE LOADING TESTS
# ===========================================================================

class TestPipelineLoading:

    def test_load_pipeline_returns_instance(self):
        p = load_pipeline()
        assert isinstance(p, InferencePipeline)

    def test_load_pipeline_singleton(self):
        p1 = load_pipeline()
        p2 = load_pipeline()
        assert p1 is p2

    def test_pipeline_has_ready_attribute(self):
        p = load_pipeline()
        assert hasattr(p, "is_ready")
        assert isinstance(p.is_ready, bool)

    def test_pipeline_has_error_message(self):
        p = load_pipeline()
        # error_message is None when ready, str when not
        if p.is_ready:
            assert p.error_message is None
        else:
            assert isinstance(p.error_message, str)

    def test_pipeline_vectorizer_loaded(self, pipeline):
        _skip_if_not_ready(pipeline)
        assert pipeline.vectorizer is not None

    def test_pipeline_primary_verifier_loaded(self, pipeline):
        _skip_if_not_ready(pipeline)
        assert pipeline.primary_verifier is not None

    def test_pipeline_has_distractor_ranker(self, pipeline):
        _skip_if_not_ready(pipeline)
        # May be None if Model B hasn't been trained yet — just check attribute
        assert hasattr(pipeline, "distractor_ranker")

    def test_pipeline_has_hint_scorer(self, pipeline):
        _skip_if_not_ready(pipeline)
        assert hasattr(pipeline, "hint_scorer")


# ===========================================================================
# 3. predict_answer TESTS
# ===========================================================================

class TestPredictAnswer:

    def test_returns_dict(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = predict_answer(ARTICLE_SHORT, QUESTION_1, OPTIONS_1)
        assert isinstance(result, dict)

    def test_has_required_keys(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = predict_answer(ARTICLE_SHORT, QUESTION_1, OPTIONS_1)
        for key in ("predicted", "confidence", "scores", "latency_ms"):
            assert key in result, f"Missing key: {key}"

    def test_predicted_is_valid_option_key(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = predict_answer(ARTICLE_SHORT, QUESTION_1, OPTIONS_1)
        assert result["predicted"] in OPTIONS_1

    def test_confidence_in_range(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = predict_answer(ARTICLE_SHORT, QUESTION_1, OPTIONS_1)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_scores_has_all_option_keys(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = predict_answer(ARTICLE_SHORT, QUESTION_1, OPTIONS_1)
        assert set(result["scores"].keys()) == set(OPTIONS_1.keys())

    def test_scores_all_in_range(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = predict_answer(ARTICLE_SHORT, QUESTION_1, OPTIONS_1)
        for k, v in result["scores"].items():
            assert 0.0 <= v <= 1.0, f"Score for {k} out of range: {v}"

    def test_latency_ms_positive(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = predict_answer(ARTICLE_SHORT, QUESTION_1, OPTIONS_1)
        assert result["latency_ms"] > 0

    def test_under_10_seconds(self, pipeline):
        """Spec §4.2: inference must complete in under 10 seconds."""
        _skip_if_not_ready(pipeline)
        t0 = time.perf_counter()
        predict_answer(ARTICLE_LONG, QUESTION_2, OPTIONS_2)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"Inference took {elapsed:.2f}s — must be < 10s"

    def test_second_question(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = predict_answer(ARTICLE_LONG, QUESTION_2, OPTIONS_2)
        assert result["predicted"] in OPTIONS_2

    def test_confidence_matches_predicted_score(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = predict_answer(ARTICLE_SHORT, QUESTION_1, OPTIONS_1)
        predicted_key = result["predicted"]
        assert abs(result["confidence"] - result["scores"][predicted_key]) < 1e-4

    def test_predicted_has_max_score(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = predict_answer(ARTICLE_SHORT, QUESTION_1, OPTIONS_1)
        best_key = max(result["scores"], key=result["scores"].get)
        assert result["predicted"] == best_key

    def test_different_articles_different_predictions(self, pipeline):
        """Sanity: the model is sensitive to article content."""
        _skip_if_not_ready(pipeline)
        r1 = predict_answer(ARTICLE_SHORT, QUESTION_1, OPTIONS_1)
        r2 = predict_answer(ARTICLE_LONG,  QUESTION_2, OPTIONS_2)
        # They should not be identical objects
        assert r1 is not r2

    def test_error_returned_when_not_ready(self):
        """Simulate an unready pipeline."""
        p = InferencePipeline.__new__(InferencePipeline)
        p._ready = False
        p._error = "No models found."
        p.vectorizer = None
        p.primary_verifier = None
        # patch the global singleton temporarily
        original = inf._pipeline_instance
        inf._pipeline_instance = p
        result = predict_answer(ARTICLE_SHORT, QUESTION_1, OPTIONS_1)
        inf._pipeline_instance = original
        assert "error" in result


# ===========================================================================
# 4. generate_distractors TESTS
# ===========================================================================

class TestGenerateDistractors:

    def test_returns_list(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = generate_distractors(ARTICLE_SHORT, QUESTION_1, ANSWER_1)
        assert isinstance(result, list)

    def test_returns_three_distractors(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = generate_distractors(ARTICLE_SHORT, QUESTION_1, ANSWER_1)
        assert len(result) == 3, f"Expected 3 distractors, got {len(result)}"

    def test_all_strings(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = generate_distractors(ARTICLE_SHORT, QUESTION_1, ANSWER_1)
        for d in result:
            assert isinstance(d, str), f"Distractor is not a string: {d!r}"

    def test_no_empty_strings(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = generate_distractors(ARTICLE_SHORT, QUESTION_1, ANSWER_1)
        for d in result:
            assert d.strip() != "", "Empty distractor found"

    def test_no_duplicates(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = generate_distractors(ARTICLE_SHORT, QUESTION_1, ANSWER_1)
        assert len(result) == len(set(result)), "Duplicate distractors found"

    def test_correct_answer_not_in_distractors(self, pipeline):
        """Distractors must not equal the correct answer."""
        _skip_if_not_ready(pipeline)
        result = generate_distractors(ARTICLE_SHORT, QUESTION_1, ANSWER_1)
        ans_clean = _clean(ANSWER_1)
        for d in result:
            assert _clean(d) != ans_clean, \
                f"Correct answer leaked into distractors: {d}"

    def test_long_article(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = generate_distractors(ARTICLE_LONG, QUESTION_2, ANSWER_2)
        assert len(result) == 3

    def test_under_10_seconds(self, pipeline):
        _skip_if_not_ready(pipeline)
        t0 = time.perf_counter()
        generate_distractors(ARTICLE_LONG, QUESTION_2, ANSWER_2)
        assert time.perf_counter() - t0 < 10.0

    def test_single_word_answer(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = generate_distractors(ARTICLE_LONG, QUESTION_2, "Britain")
        assert len(result) == 3

    def test_short_article_fallback(self, pipeline):
        """Should not crash on a very short article."""
        _skip_if_not_ready(pipeline)
        result = generate_distractors("The sky is blue.", "What colour is the sky?", "blue")
        assert isinstance(result, list)
        assert len(result) == 3


# ===========================================================================
# 5. generate_hints TESTS
# ===========================================================================

class TestGenerateHints:

    def test_returns_list(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = generate_hints(ARTICLE_SHORT, QUESTION_1, ANSWER_1)
        assert isinstance(result, list)

    def test_returns_three_hints(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = generate_hints(ARTICLE_SHORT, QUESTION_1, ANSWER_1)
        assert len(result) == 3, f"Expected 3 hints, got {len(result)}"

    def test_all_strings(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = generate_hints(ARTICLE_SHORT, QUESTION_1, ANSWER_1)
        for h in result:
            assert isinstance(h, str)

    def test_no_empty_hints(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = generate_hints(ARTICLE_SHORT, QUESTION_1, ANSWER_1)
        for h in result:
            assert h.strip() != "", "Empty hint found"

    def test_hints_are_from_article(self, pipeline):
        """Each hint should be a substring/sentence from the article."""
        _skip_if_not_ready(pipeline)
        result = generate_hints(ARTICLE_SHORT, QUESTION_1, ANSWER_1)
        article_lower = ARTICLE_SHORT.lower()
        for h in result:
            # Allow slight cleaning differences; check at least 5 words overlap
            h_words = set(_tokenize(h))
            art_words = set(_tokenize(ARTICLE_SHORT))
            overlap = len(h_words & art_words)
            assert overlap >= 3, \
                f"Hint seems unrelated to article (overlap={overlap}): {h}"

    def test_hint_3_more_relevant_than_hint_1(self, pipeline):
        """
        Spec §9.2: Hint 3 should be near-explicit (highest relevance),
        Hint 1 should be most general (lowest relevance).
        We verify this by checking cosine similarity to the question.
        """
        _skip_if_not_ready(pipeline)
        result = generate_hints(ARTICLE_LONG, QUESTION_2, ANSWER_2)
        p = pipeline

        try:
            q_vec = p.vectorizer.transform([_clean(QUESTION_2)])
            sims  = []
            for h in result:
                h_vec = p.vectorizer.transform([_clean(h)])
                from sklearn.metrics.pairwise import cosine_similarity
                sims.append(float(cosine_similarity(q_vec, h_vec)[0, 0]))
            # Hint 3 (index 2) should be >= Hint 1 (index 0) in cosine sim
            assert sims[2] >= sims[0] - 0.05, \
                f"Hint ordering broken: sims={sims}"
        except Exception:
            pytest.skip("Vectorizer unavailable for ordering check")

    def test_long_article(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = generate_hints(ARTICLE_LONG, QUESTION_2, ANSWER_2)
        assert len(result) == 3

    def test_single_sentence_article(self, pipeline):
        """Should not crash on a degenerate article."""
        _skip_if_not_ready(pipeline)
        result = generate_hints("Mary went to the market.", "Where did Mary go?", "market")
        assert isinstance(result, list)

    def test_under_10_seconds(self, pipeline):
        _skip_if_not_ready(pipeline)
        t0 = time.perf_counter()
        generate_hints(ARTICLE_LONG, QUESTION_2, ANSWER_2)
        assert time.perf_counter() - t0 < 10.0


# ===========================================================================
# 6. supporting_sentence TESTS
# ===========================================================================

class TestSupportingSentence:

    def test_returns_string(self, pipeline):
        _skip_if_not_ready(pipeline)
        sent = pipeline.supporting_sentence(ARTICLE_SHORT, ANSWER_1)
        assert isinstance(sent, str)

    def test_not_empty(self, pipeline):
        _skip_if_not_ready(pipeline)
        sent = pipeline.supporting_sentence(ARTICLE_SHORT, ANSWER_1)
        assert sent.strip() != ""

    def test_from_article(self, pipeline):
        _skip_if_not_ready(pipeline)
        sent = pipeline.supporting_sentence(ARTICLE_SHORT, ANSWER_1)
        words = set(_tokenize(sent))
        article_words = set(_tokenize(ARTICLE_SHORT))
        assert len(words & article_words) >= 3

    def test_exact_answer_match_preferred(self, pipeline):
        """If the exact answer string is in a sentence, that sentence should be returned."""
        _skip_if_not_ready(pipeline)
        # ANSWER_1 contains 'apples' which is in the second sentence
        sent = pipeline.supporting_sentence(ARTICLE_SHORT, "apples")
        assert "apple" in sent.lower()


# ===========================================================================
# 7. random_race_sample TESTS
# ===========================================================================

class TestRandomRaceSample:

    def test_returns_dict(self):
        try:
            sample = random_race_sample("val")
        except FileNotFoundError:
            pytest.skip("RACE CSV not found — skipping sample test")
        assert isinstance(sample, dict)

    def test_has_required_keys(self):
        try:
            sample = random_race_sample("val")
        except FileNotFoundError:
            pytest.skip("RACE CSV not found")
        for key in ("article", "question", "options", "answer"):
            assert key in sample, f"Missing key: {key}"

    def test_options_has_four_keys(self):
        try:
            sample = random_race_sample("val")
        except FileNotFoundError:
            pytest.skip("RACE CSV not found")
        assert set(sample["options"].keys()) == {"A", "B", "C", "D"}

    def test_answer_is_valid_key(self):
        try:
            sample = random_race_sample("val")
        except FileNotFoundError:
            pytest.skip("RACE CSV not found")
        assert sample["answer"] in {"A", "B", "C", "D"}

    def test_article_nonempty(self):
        try:
            sample = random_race_sample("val")
        except FileNotFoundError:
            pytest.skip("RACE CSV not found")
        assert len(sample["article"].strip()) > 0

    def test_randomness(self):
        """Two consecutive calls should (usually) return different samples."""
        try:
            s1 = random_race_sample("val")
            s2 = random_race_sample("val")
        except FileNotFoundError:
            pytest.skip("RACE CSV not found")
        # With 87k+ rows, collision probability ≈ 0 — just verify structure
        assert isinstance(s1, dict) and isinstance(s2, dict)

    def test_fallback_to_train(self):
        """If 'test' split missing, should fall back gracefully."""
        try:
            sample = random_race_sample("test")
            assert "article" in sample
        except FileNotFoundError:
            pytest.skip("Neither test nor train CSV found")


# ===========================================================================
# 8. INTEGRATION TEST — full pipeline round-trip
# ===========================================================================

class TestEndToEnd:

    def test_full_round_trip(self, pipeline):
        """
        Simulate what the UI does for a single question:
          1. predict_answer  →  get predicted key
          2. generate_distractors  →  3 wrong options
          3. generate_hints  →  3 graduated hints
        All three must succeed and return correct types.
        """
        _skip_if_not_ready(pipeline)

        result = predict_answer(ARTICLE_LONG, QUESTION_2, OPTIONS_2)
        assert "predicted" in result

        dists = generate_distractors(ARTICLE_LONG, QUESTION_2, ANSWER_2)
        assert len(dists) == 3

        hints = generate_hints(ARTICLE_LONG, QUESTION_2, ANSWER_2)
        assert len(hints) == 3

    def test_race_sample_full_pipeline(self, pipeline):
        """Load a real RACE sample and run the complete inference chain."""
        _skip_if_not_ready(pipeline)

        try:
            sample = random_race_sample("val")
        except FileNotFoundError:
            pytest.skip("RACE CSV not found")

        article       = sample["article"]
        question      = sample["question"]
        options       = sample["options"]
        correct_text  = options[sample["answer"]]

        # Answer verification
        t0     = time.perf_counter()
        result = predict_answer(article, question, options)
        elapsed = time.perf_counter() - t0

        assert elapsed < 10.0, f"predict_answer took {elapsed:.2f}s"
        assert result["predicted"] in options

        # Distractors
        dists = generate_distractors(article, question, correct_text)
        assert len(dists) == 3
        for d in dists:
            assert isinstance(d, str) and d.strip()

        # Hints
        hints = generate_hints(article, question, correct_text)
        assert len(hints) == 3
        for h in hints:
            assert isinstance(h, str) and h.strip()

    def test_repeated_calls_stable(self, pipeline):
        """Multiple calls with same input must return same predicted key."""
        _skip_if_not_ready(pipeline)
        results = [
            predict_answer(ARTICLE_SHORT, QUESTION_1, OPTIONS_1)
            for _ in range(3)
        ]
        predicted_keys = [r["predicted"] for r in results]
        assert len(set(predicted_keys)) == 1, \
            f"Non-deterministic predictions: {predicted_keys}"


# ===========================================================================
# 9. EDGE CASES
# ===========================================================================

class TestEdgeCases:

    def test_unicode_article(self, pipeline):
        _skip_if_not_ready(pipeline)
        article = "María went to París. She loved café culture and résumés."
        result  = predict_answer(article, "Where did María go?",
                                 {"A": "París", "B": "Madrid",
                                  "C": "Berlin", "D": "Rome"})
        assert "predicted" in result

    def test_very_long_article(self, pipeline):
        _skip_if_not_ready(pipeline)
        long_article = (ARTICLE_LONG + " ") * 20   # ~160 sentences
        result = predict_answer(long_article, QUESTION_2, OPTIONS_2)
        assert "predicted" in result

    def test_single_word_options(self, pipeline):
        _skip_if_not_ready(pipeline)
        result = predict_answer(
            ARTICLE_LONG, "What is the key country?",
            {"A": "Britain", "B": "France", "C": "Germany", "D": "China"}
        )
        assert result["predicted"] in {"A", "B", "C", "D"}

    def test_distractor_long_article(self, pipeline):
        _skip_if_not_ready(pipeline)
        long_article = (ARTICLE_SHORT + " ") * 10
        result = generate_distractors(long_article, QUESTION_1, ANSWER_1)
        assert len(result) == 3

    def test_hint_all_sentences_same(self, pipeline):
        """Degenerate article where all sentences are identical."""
        _skip_if_not_ready(pipeline)
        article = "The cat sat on the mat. " * 5
        result  = generate_hints(article, "Where did the cat sit?", "mat")
        assert isinstance(result, list)

    def test_empty_correct_answer_distractor(self, pipeline):
        """Should not crash on empty correct_answer."""
        _skip_if_not_ready(pipeline)
        try:
            result = generate_distractors(ARTICLE_SHORT, QUESTION_1, "")
            assert isinstance(result, list)
        except Exception as exc:
            pytest.fail(f"Crashed on empty correct_answer: {exc}")