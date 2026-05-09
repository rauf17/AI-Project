
## 1. Project Overview

This project implements an **AI-powered Reading Comprehension and Quiz Generation System** using the RACE dataset. The system is composed of two independent ML pipelines wired together through a polished multi-screen UI:

| Component | Responsibility |
|---|---|
| **Model A** | Question & Answer Generator / Verifier (Traditional ML + Unsupervised) |
| **Model B** | Distractor & Hint Generator (Traditional ML) |
| **UI Layer** | Interactive multi-screen application (Streamlit / PyQt5 / React) |

### What the system does end-to-end

1. User pastes a reading passage (or loads a random RACE sample).
2. Model A generates a multiple-choice question and identifies the correct answer.
3. Model B generates three plausible distractor options and graduated hints.
4. The UI presents the full quiz; the user selects an answer.
5. Model A verifies the user's choice and displays colour-coded feedback.
6. The Analytics Dashboard logs all inferences with performance metrics.

---

## 2. Dataset

### 2.1 RACE Dataset

**RACE** (ReAding Comprehension from Examinations) was published by Lai et al. (2017). It contains English reading passages and multiple-choice questions extracted from Chinese middle-school and high-school English examinations (age range 12–18).

| Metric | Value |
|---|---|
| Total Passages | 28,000+ |
| Total Questions | ~100,000 |
| Question Types | Multiple-choice (A / B / C / D) |
| Source | Chinese middle & high school English exams |
| Language | English |
| Loading Method | `pandas.read_csv()` |
| Split | Train / Dev / Test |

**Download:** [Kaggle — RACE Dataset](https://www.kaggle.com/datasets/swaptr/race-dataset)

Place the files as follows after downloading:

```
data/raw/train.csv
data/raw/val.csv
data/raw/test.csv
```

### 2.2 Dataset Column Schema

| Column | Type | Description |
|---|---|---|
| `id` | String | Unique identifier for the entry |
| `article` | String | Full reading passage (paragraph-level text) |
| `question` | String | Multiple-choice question related to the article |
| `A` | String | Answer option A |
| `B` | String | Answer option B |
| `C` | String | Answer option C |
| `D` | String | Answer option D |
| `answer` | String | Correct answer label (A, B, C, or D) |

### 2.3 Loading the Dataset

```python
import pandas as pd

train_df = pd.read_csv('data/raw/train.csv')
val_df   = pd.read_csv('data/raw/val.csv')
test_df  = pd.read_csv('data/raw/test.csv')

print(train_df.shape)    # Expected: ~87,866 rows
print(train_df.columns)  # id, article, question, A, B, C, D, answer
```

---

## 3. Project Structure

```
race_rc_project/
├── data/
│   ├── raw/                        # Original RACE CSV files (train/val/test)
│   └── processed/                  # Tokenized, feature-engineered data
│       ├── X_train_onehot.npz      # Sparse One-Hot feature matrix (train)
│       ├── X_val_onehot.npz        # Sparse One-Hot feature matrix (val)
│       ├── X_test_onehot.npz       # Sparse One-Hot feature matrix (test)
│       ├── X_train_tfidf.npz       # [Optional] TF-IDF feature matrix
│       ├── y_train.npy             # Labels (1 = correct option, 0 = incorrect)
│       └── cosine_features.npz     # Handcrafted cosine similarity features
├── models/
│   ├── model_a/
│   │   └── traditional/
│   │       ├── lr_classifier.pkl           # Logistic Regression
│   │       ├── svm_classifier.pkl          # Support Vector Machine
│   │       ├── nb_classifier.pkl           # Naive Bayes
│   │       ├── rf_classifier.pkl           # Random Forest
│   │       ├── xgb_classifier.pkl          # XGBoost
│   │       ├── ensemble_meta.pkl           # Stacking meta-classifier
│   │       ├── onehot_encoder.pkl          # Fitted OneHotEncoder
│   │       └── tfidf_vectorizer.pkl        # [Optional] Fitted TfidfVectorizer
│   └── model_b/
│       └── traditional/
│           ├── distractor_ranker.pkl       # LR / RF distractor ranker
│           ├── hint_scorer.pkl             # LR hint sentence scorer
│           └── word2vec_model.bin          # Pre-trained / fine-tuned Word2Vec
├── src/
│   ├── preprocessing.py            # Dataset loading & ALL feature engineering
│   ├── model_a_train.py            # Full training script for Model A
│   ├── model_b_train.py            # Full training script for Model B
│   ├── inference.py                # Unified inference API (used by UI)
│   └── evaluate.py                 # All metric computation functions
├── ui/
│   ├── app.py                      # Streamlit / PyQt5 / FastAPI entry point
│   └── components/                 # Reusable UI components
├── notebooks/
│   ├── EDA.ipynb                   # Exploratory Data Analysis (required)
│   └── experiments.ipynb           # Experiment tracking
├── tests/
│   └── test_inference.py           # Unit tests for inference pipeline
├── requirements.txt                # All dependencies with pinned versions
├── README.md                       # This file
└── report/
    └── final_report.pdf
```

---

## 4. Environment Setup

### 4.1 Requirements

```
python>=3.9
pandas==2.1.4
numpy==1.26.3
scikit-learn==1.4.0
scipy==1.12.0
xgboost==2.0.3
lightgbm==4.2.0
gensim==4.3.2
sentence-transformers==2.3.1
joblib==1.3.2
streamlit==1.31.0      # UI option 1
PyQt5==5.15.10         # UI option 2
fastapi==0.109.2       # UI option 3
uvicorn==0.27.0        # UI option 3
matplotlib==3.8.2
seaborn==0.13.2
plotly==5.18.0
jupyter==1.0.0
pytest==7.4.4
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

### 4.2 GPU / Hardware Requirement

All models must be trainable on:
- NVIDIA RTX 3060 (12 GB VRAM), **or**
- Google Colab T4 (free tier)

Do **not** use architectures requiring > 24 GB VRAM. Inference for a single article + question must complete in **under 10 seconds**.

---

## 5. Preprocessing Pipeline

All preprocessing logic lives in `src/preprocessing.py`. Run it once to generate all processed feature matrices before any training.

```bash
python src/preprocessing.py
```

### 5.1 What Preprocessing Must Do

#### Step 1 — Load Raw Data

```python
train_df = pd.read_csv('data/raw/train.csv')
val_df   = pd.read_csv('data/raw/val.csv')
test_df  = pd.read_csv('data/raw/test.csv')
```

#### Step 2 — Text Cleaning

Apply to `article`, `question`, and all four option columns (A, B, C, D):

- Convert all text to **lowercase**
- Remove **punctuation** (keep apostrophes for contractions if desired)
- Strip leading/trailing whitespace
- Collapse multiple spaces into one
- Optionally remove digits (configurable flag)

```python
import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

#### Step 3 — Build the Verification Label Matrix

For each row in the dataset, expand to **4 rows** (one per option). Label = 1 if the option matches `answer`, else 0. This creates the binary classification target for Model A.

```python
# For each row: 4 samples (one per option A/B/C/D)
# Label: 1 if this option is the correct answer, else 0
# y_train.mean() should be close to 0.25
```

Expected class balance: `y_train.mean() ≈ 0.25` (one correct per four options).

#### Step 4 — One-Hot Encoding (Primary Feature Representation)

One-Hot Encoding is the **primary** feature representation for this project. TF-IDF is optional.

**4a. Build vocabulary** from the training corpus (articles + questions + options):

```python
from sklearn.feature_extraction.text import CountVectorizer

# Build vocabulary using CountVectorizer
vocab_vectorizer = CountVectorizer(
    max_features=10000,
    stop_words='english',
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 1)
)
vocab_vectorizer.fit(train_corpus)
```

**4b. Binarize** — convert counts to binary presence/absence (0 or 1):

```python
from sklearn.preprocessing import Binarizer

# Each token is either present (1) or absent (0)
X_train_counts = vocab_vectorizer.transform(train_corpus)
binarizer = Binarizer()
X_train_onehot = binarizer.fit_transform(X_train_counts)
```

**4c. Combined feature string** for each sample:
```
combined = article + " " + article + " " + question + " " + option_text
```
(Article is repeated to give it more weight relative to the short option text.)

**4d. Save** fitted encoder and matrices:

```python
import joblib
from scipy import sparse

joblib.dump(vocab_vectorizer, 'models/model_a/traditional/onehot_encoder.pkl')
sparse.save_npz('data/processed/X_train_onehot.npz', X_train_onehot)
sparse.save_npz('data/processed/X_val_onehot.npz', X_val_onehot)
sparse.save_npz('data/processed/X_test_onehot.npz', X_test_onehot)
```

> ⚠️ **Critical:** Call `fit_transform()` only on training data. Use `transform()` on val and test to avoid data leakage.

#### Step 5 — TF-IDF Vectorization (Optional)

TF-IDF is an **optional** alternative or supplement to One-Hot Encoding.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    sublinear_tf=True,       # log(1+TF) — dampens high-frequency terms
    ngram_range=(1, 2),      # unigrams + bigrams
    min_df=2,
    max_df=0.95,
    norm='l2'                # required for cosine similarity
)

X_train_tfidf = vectorizer.fit_transform(train_corpus)
X_val_tfidf   = vectorizer.transform(val_corpus)
X_test_tfidf  = vectorizer.transform(test_corpus)

joblib.dump(vectorizer, 'models/model_a/traditional/tfidf_vectorizer.pkl')
sparse.save_npz('data/processed/X_train_tfidf.npz', X_train_tfidf)
```

#### Step 6 — Cosine Similarity Features (Handcrafted)

Compute cosine similarity scores between pairs of text fields. These become additional numerical features appended to the One-Hot (or TF-IDF) matrix.

```python
from sklearn.metrics.pairwise import cosine_similarity

# Feature 1: cosine(article, option)
# Feature 2: cosine(question, option)
# Feature 3: cosine(article, question)
# Feature 4: character-level overlap (correct_answer vs candidate)
# Feature 5: word length ratio (len(option) / len(article))
# Feature 6: passage frequency of option keywords
```

Concatenate cosine features with the sparse One-Hot matrix using `scipy.sparse.hstack`.

#### Step 7 — Handcrafted Lexical Features

Additional numerical features for Model A and B:

- Keyword overlap count between question and each option
- Option length (word count)
- Article length (sentence count)
- Position of matching sentence in article (normalized 0–1)
- Jaccard similarity between question tokens and option tokens
- Number of named entities in option (via simple capitalized word count)

#### Step 8 — Save All Processed Outputs

```
data/processed/X_train_onehot.npz
data/processed/X_val_onehot.npz
data/processed/X_test_onehot.npz
data/processed/X_train_tfidf.npz       # optional
data/processed/cosine_features.npz
data/processed/y_train.npy
data/processed/y_val.npy
data/processed/y_test.npy
```

---

## 6. Model A — Question & Answer Generator / Verifier

### 6.1 What Model A Does

Model A performs two tasks:

1. **Generation task:** given an article, extract a candidate sentence and apply Wh-word templates to produce a multiple-choice question. Identify the correct answer span from the passage.
2. **Verification task:** given an article, a question, and one option, predict whether that option is correct (binary classification).

Train with:

```bash
python src/model_a_train.py
```

---

### 6.2 Supervised Models for Answer Verification

Students must implement **at least two** of the following models and compare them:

#### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs')
lr.fit(X_train, y_train)
joblib.dump(lr, 'models/model_a/traditional/lr_classifier.pkl')
```

- **Features:** One-Hot Encoding of (article + question + option). Cosine similarity features appended.
- **Task:** Answer verification (binary: correct / incorrect)
- **Report:** Accuracy, Macro F1, Precision, Recall, Confusion Matrix on val set.

#### Support Vector Machine (SVM)

```python
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

svm = LinearSVC(C=1.0, class_weight='balanced', max_iter=2000)
svm_cal = CalibratedClassifierCV(svm)   # for probability output
svm_cal.fit(X_train, y_train)
joblib.dump(svm_cal, 'models/model_a/traditional/svm_classifier.pkl')
```

- **Features:** One-Hot Encoding + cosine similarity features
- **Task:** Answer verification & ranking
- **Note:** Use `LinearSVC` (faster on sparse data) over kernel SVM for RACE-scale data.

#### Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB(alpha=1.0)
nb.fit(X_train_counts, y_train)   # requires non-negative counts (not cosine features)
joblib.dump(nb, 'models/model_a/traditional/nb_classifier.pkl')
```

- **Features:** Bag-of-words counts from question tokens only
- **Task:** Question type classification (factual / inferential / vocabulary / etc.)

#### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
rf.fit(X_train_lexical, y_train)  # use handcrafted lexical features only
joblib.dump(rf, 'models/model_a/traditional/rf_classifier.pkl')
```

- **Features:** Handcrafted lexical features (keyword overlap, length ratios, position features)
- **Task:** Difficulty estimation

#### XGBoost

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                    use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
xgb.fit(X_train_dense, y_train, eval_set=[(X_val_dense, y_val)], early_stopping_rounds=20)
joblib.dump(xgb, 'models/model_a/traditional/xgb_classifier.pkl')
```

- **Features:** Semantic similarity + length features (dense)
- **Task:** Answer verification

---

### 6.3 Unsupervised & Semi-Supervised Approaches (Required)

Students must implement **at least one** of the following:

#### K-Means Clustering

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_train_onehot)

# Evaluate: clustering purity, silhouette score
from sklearn.metrics import silhouette_score
sil = silhouette_score(X_train_onehot, kmeans.labels_, sample_size=5000)
```

- Group question-answer pairs by One-Hot Encoded feature similarity.
- Goal: discover latent answer patterns without labels.
- Report: clustering purity, silhouette score. Compare against supervised baselines.

#### Label Propagation (Semi-Supervised)

```python
from sklearn.semi_supervised import LabelPropagation

# Use a small labeled set; -1 = unlabeled
lp = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=1000)
lp.fit(X_semi, y_semi_partial)  # y_semi_partial has -1 for unlabeled
```

- Use a small labeled set to propagate labels to nearby unlabeled samples in feature space.
- Report: semi-supervised F1 vs. fully supervised F1.

#### Gaussian Mixture Models (GMM)

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(X_train_dense_reduced)  # use PCA-reduced dense features
```

- Probabilistic clustering assigning soft membership to question-answer clusters.
- Useful for identifying question type patterns.
- Report: BIC score, cluster-to-label alignment table.

---

### 6.4 Template-Based Question Generation

The generation sub-task uses rule-based templates + ML ranker:

**Step 1 — Candidate Sentence Extraction**

```python
# Score each sentence in article by One-Hot keyword overlap with correct answer
# Select top-scoring sentence as question stem candidate
def extract_candidate_sentences(article, answer, vectorizer):
    sentences = article.split('.')
    answer_vec = vectorizer.transform([answer])
    scores = []
    for sent in sentences:
        sent_vec = vectorizer.transform([sent])
        sim = cosine_similarity(answer_vec, sent_vec)[0][0]
        scores.append(sim)
    return sorted(zip(scores, sentences), reverse=True)
```

**Step 2 — Wh-word Template Application**

```python
WH_TEMPLATES = {
    'person':    "Who {verb} {rest}?",
    'place':     "Where {verb} {rest}?",
    'time':      "When {verb} {rest}?",
    'reason':    "Why {verb} {rest}?",
    'thing':     "What {verb} {rest}?",
}
# Apply appropriate template based on answer type (person/place/time/object)
# Answer type detection: simple keyword heuristics (no NLP library required)
```

**Step 3 — ML Question Ranker**

Train an SVM or Random Forest to score generated questions by fluency and relevance features:

- Question length (word count)
- Wh-word type
- Number of content words in question overlapping with article
- Sentence position in article (normalized)

---

### 6.5 Model A Evaluation Metrics

| Metric | Application |
|---|---|
| **Accuracy** | Fraction of correctly labeled options on the verification task |
| **Macro F1** | Verification task — handles class imbalance across option labels |
| **Exact Match (EM)** | Strict character-level match between predicted and gold answer label |
| **Precision** | Fraction of predicted-correct options that are truly correct |
| **Recall** | Fraction of truly correct options that were predicted correct |
| **Confusion Matrix** | 2×2 matrix: True Positives / False Positives / True Negatives / False Negatives |

---

## 7. Model B — Distractor & Hint Generator

### 7.1 What Model B Does

Given a reading passage, a question, and the correct answer, Model B produces:

1. **Three plausible distractors** — wrong answer options that are semantically related but factually incorrect.
2. **Graduated hints** — three increasingly specific hints that guide the reader toward the correct answer without revealing it directly.

Train with:

```bash
python src/model_b_train.py
```

---

### 7.2 Distractor Generation Requirements

Good distractors must satisfy **all four** constraints simultaneously:

| Constraint | Description |
|---|---|
| **Plausibility** | Must look like legitimate answers to an uninformed reader |
| **Incorrectness** | Must be definitively wrong with respect to the passage |
| **Diversity** | The three distractors must not be trivially similar to each other |
| **Grammatical consistency** | All options must share the same syntactic form |

---

### 7.3 Supervised Distractor Ranking Pipeline

**Step 1 — Candidate Extraction**

```python
def extract_candidates(passage, answer):
    # Retrieve all candidate phrases from the passage
    # Use simple string matching and frequency-based word selection
    # No external NLP tools required
    tokens = passage.lower().split()
    freq = Counter(tokens)
    candidates = [w for w, c in freq.most_common(50) if w not in stopwords and w != answer.lower()]
    return candidates
```

**Step 2 — Feature Engineering per Candidate**

For each candidate distractor compute:

- One-Hot cosine similarity to the correct answer (primary)
- TF-IDF cosine similarity to the correct answer (optional)
- Character-level match score between candidate and answer
- Passage frequency of the candidate
- Position of first occurrence in passage (normalized)
- Word length difference from correct answer

**Step 3 — ML Ranker**

Train a Logistic Regression or Random Forest classifier. Label = 1 if the candidate is one of the original dataset's distractor options, else 0.

```python
from sklearn.linear_model import LogisticRegression

ranker = LogisticRegression(C=1.0, max_iter=1000)
ranker.fit(X_distractor_features, y_distractor_labels)

# Select top-3 non-answer candidates as final distractors
joblib.dump(ranker, 'models/model_b/traditional/distractor_ranker.pkl')
```

---

### 7.4 Alternative: One-Hot Vocabulary + Cosine Similarity

```python
# Build One-Hot vocabulary over full training corpus
# For each answer token, retrieve top-N most co-occurring terms as synonym candidates
# Re-rank by cosine similarity using One-Hot representations
# Apply diversity penalty: penalize candidates too similar to already-selected distractors
diversity_threshold = 0.7  # reject if cosine_sim > threshold with any already-chosen distractor
```

---

### 7.5 Word2Vec Nearest Neighbours (Alternative)

```python
import gensim.downloader as api

w2v_model = api.load('word2vec-google-news-300')

def get_word2vec_distractors(answer, passage, top_n=10):
    neighbours = w2v_model.most_similar(answer.lower(), topn=top_n)
    # Filter out candidates that appear verbatim in the passage
    distractors = [w for w, _ in neighbours if w.lower() not in passage.lower()]
    return distractors[:3]
```

---

### 7.6 Extractive Hint Generation

Hints are extracted as the most relevant passage sentences to the question.

**Extractive Strategy:**

```python
def generate_hints(article, question, vectorizer, n_hints=3):
    sentences = [s.strip() for s in article.split('.') if len(s.strip()) > 10]
    question_vec = vectorizer.transform([question])
    scores = []
    for i, sent in enumerate(sentences):
        sent_vec = vectorizer.transform([sent])
        sim = cosine_similarity(question_vec, sent_vec)[0][0]
        position_penalty = i / len(sentences)   # earlier sentences slightly preferred
        scores.append(sim - 0.05 * position_penalty)
    ranked = sorted(zip(scores, sentences), reverse=True)
    # Return hints from most general → most specific
    # Hint 1: lowest relevance sentence (general context)
    # Hint 2: middle relevance
    # Hint 3: highest relevance (near-explicit)
    hints = [sent for _, sent in ranked[:n_hints]]
    return hints[::-1]  # reverse so Hint 1 is most general
```

**ML-Scored Hint Strategy:**

Train a Logistic Regression on sentence features (no neural network):

```python
# Features per sentence:
# - keyword overlap count (question tokens ∩ sentence tokens)
# - sentence position in article (normalized)
# - sentence length (word count)
# - cosine similarity to question (One-Hot based)
# Label: 1 if this sentence contains the answer span, else 0

hint_scorer = LogisticRegression(C=1.0)
hint_scorer.fit(X_sentence_features, y_contains_answer)
joblib.dump(hint_scorer, 'models/model_b/traditional/hint_scorer.pkl')
```

**Hint Graduation Rules:**

- **Hint 1 (most general):** passage sentence with the lowest but still positive relevance score — gives broad context only.
- **Hint 2 (more specific):** sentence with medium relevance score — narrows the topic.
- **Hint 3 (near-explicit):** highest-scoring sentence — strongly implies the answer.
- **'Reveal Answer'** button only appears after the user has consumed all three hints.

---

### 7.7 Frequency-Based Substitution (Alternative Distractor Method)

```python
# Identify high-frequency content words in passage
# (nouns and proper nouns identified via capitalization + frequency counting)
# Substitute correct answer with similarly frequent but semantically distinct terms
# Ensure substituted terms do NOT appear in close proximity to the question in the passage
```

---

### 7.8 Model B Evaluation Metrics

| Metric | Application |
|---|---|
| **Precision** | Fraction of generated distractors that are plausible but not the correct answer |
| **Recall** | Fraction of reference dataset distractors recovered by the pipeline |
| **Macro F1** | Harmonic mean of Precision and Recall for distractor ranking |
| **Accuracy** | Fraction of test samples where top-ranked distractor candidate is NOT the correct answer |
| **Confusion Matrix** | Human evaluation: 1–5 Likert scale rating of distractor believability |
| **Precision@K** | Fraction of top-K ranked hint sentences overlapping with the gold key sentence |
| **R² Score** | For regression-based hint scorer: correlation between predicted and true sentence relevance |

---

## 8. Ensemble Strategy

Students are encouraged to combine multiple Model A classifiers into an ensemble. Implement at least one of:

### Soft Voting

Average probability outputs from LR, SVM, and Naive Bayes:

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[('lr', lr), ('svm', svm_cal), ('nb', nb)],
    voting='soft'
)
ensemble.fit(X_train, y_train)
joblib.dump(ensemble, 'models/model_a/traditional/ensemble_soft.pkl')
```

### Hard Voting

Majority vote across N classifiers:

```python
ensemble_hard = VotingClassifier(
    estimators=[('lr', lr), ('svm', svm_cal), ('rf', rf)],
    voting='hard'
)
```

### Stacking

Train a meta-classifier on the out-of-fold predictions of base models:

```python
from sklearn.ensemble import StackingClassifier

stack = StackingClassifier(
    estimators=[('lr', lr), ('svm', svm_cal), ('nb', nb)],
    final_estimator=LogisticRegression(C=1.0),
    cv=5
)
stack.fit(X_train, y_train)
joblib.dump(stack, 'models/model_a/traditional/ensemble_meta.pkl')
```

Report the improvement of the ensemble over the best individual model on the test set.

---

## 9. User Interface

### 9.1 Platform Options

| Platform | Language | Best For | Difficulty |
|---|---|---|---|
| **Streamlit** ✅ Recommended | Python | Rapid ML prototypes | Easy |
| Gradio | Python | Model demos, sharing | Easy |
| PyQt5 | Python | Desktop native app | Medium |
| React + Flask | JS + Python | Full-stack web app | Hard |
| Tkinter | Python | Lightweight desktop | Easy |
| FastAPI + Jinja2 | Python | Web, minimal JS | Medium |

Run the UI (Streamlit example):

```bash
streamlit run ui/app.py
```

---

### 9.2 Required Screens

#### Screen 1 — Article Input

- Text area for pasting or uploading a reading passage
- Button to **load a random RACE sample** for quick testing
- **Submit** button that triggers both Model A and Model B inference simultaneously
- Loading spinner shown during inference

#### Screen 2 — Question & Answer Quiz View

- Displays the generated (or RACE original) question
- Shows four options (A, B, C, D): one correct answer + three distractors from Model B
- User selects an answer and clicks **Check** — Model A verifier confirms correctness
- Colour-coded result: **green** = correct, **red** = incorrect
- Explanation shown below result (which sentence from article supports the answer)

#### Screen 3 — Hint Panel

- Collapsible or tabbed panel showing graduated hints from Model B
- **Hint 1:** most general clue (broad context, does not reveal answer)
- **Hint 2:** more specific clue (narrows the topic)
- **Hint 3:** near-explicit clue (strongly implies the answer)
- **Reveal Answer** button appears only after the user has used all three hints

#### Screen 4 — Developer / Analytics Dashboard

- **Model A performance:** Accuracy, F1-Score, Precision, Recall, Confusion Matrix on the last N inferences
- **Model B performance:** Precision, Recall, F1-Score, Accuracy for distractor ranking
- Inference time per request (latency tracking in milliseconds)
- Session log table with export to CSV button
- Plots: Confusion matrix heatmap, F1 bar chart across models

---

### 9.3 UX Requirements

- Application must be usable without reading a manual (self-explanatory UI)
- All error states (empty input, model failure, unsupported file) must display a friendly message
- Loading indicators must be shown during every model inference call
- Sufficient colour contrast (WCAG AA minimum)
- Readable font sizes (minimum 14px body text)
- Keyboard navigation supported

---

## 10. Evaluation & Metrics

All evaluation code lives in `src/evaluate.py`.

```bash
python src/evaluate.py --model_a --model_b --split test
```

### 10.1 Model A Metrics

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

accuracy  = accuracy_score(y_true, y_pred)
macro_f1  = f1_score(y_true, y_pred, average='macro')
precision = precision_score(y_true, y_pred, average='macro')
recall    = recall_score(y_true, y_pred, average='macro')
cm        = confusion_matrix(y_true, y_pred)

# Exact Match: strict string comparison of predicted label vs gold label
exact_match = sum(p == g for p, g in zip(y_pred_labels, y_gold_labels)) / len(y_gold_labels)
```

### 10.2 Model B Metrics

```python
# Distractor precision/recall/F1 (binary: plausible-but-wrong vs. correct)
# R² for regression-based hint scorer
from sklearn.metrics import r2_score
r2 = r2_score(y_relevance_true, y_relevance_pred)
```

### 10.3 Unsupervised Metrics (Model A clustering)

```python
from sklearn.metrics import silhouette_score

sil_score = silhouette_score(X, cluster_labels, sample_size=5000)

# Clustering purity
def purity_score(y_true, y_pred):
    contingency = confusion_matrix(y_true, y_pred)
    return contingency.max(axis=0).sum() / contingency.sum()
```

### 10.4 Hyperparameter Tuning

Use `GridSearchCV` wrapped in a `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('enc', CountVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000)),
])

param_grid = {
    'enc__max_features': [5000, 10000, 15000],
    'enc__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.1, 1.0, 10.0],
}

gs = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=2)
gs.fit(train_texts, y_train)
print('Best params:', gs.best_params_)
print('Best F1:', gs.best_score_)
```

---

## 11. Training — End-to-End Reproduction

Run the full pipeline in order:

```bash
# Step 1: Preprocessing
python src/preprocessing.py

# Step 2: Train Model A (supervised + unsupervised)
python src/model_a_train.py

# Step 3: Train Model B (distractor + hint)
python src/model_b_train.py

# Step 4: Evaluate all models on test set
python src/evaluate.py --split test

# Step 5: Launch UI
streamlit run ui/app.py
```

Alternatively, run the end-to-end notebook:

```bash
jupyter notebook notebooks/experiments.ipynb
```

---

## 12. Inference

`src/inference.py` exposes a clean API used by the UI layer:

```python
from src.inference import predict_answer, generate_distractors, generate_hints

# Answer verification
result = predict_answer(
    article="...",
    question="What did the student do?",
    options={"A": "Studied", "B": "Slept", "C": "Played", "D": "Ate"}
)
# Returns: {"predicted": "A", "confidence": 0.84, "latency_ms": 47}

# Distractor generation
distractors = generate_distractors(
    article="...",
    question="What did the student do?",
    correct_answer="Studied hard"
)
# Returns: ["Played football", "Watched TV", "Stayed home"]

# Hint generation
hints = generate_hints(
    article="...",
    question="What did the student do?",
    correct_answer="Studied hard"
)
# Returns: [hint1_general, hint2_specific, hint3_near_explicit]
```

---

## 13. Implementation Milestones

| Week | Days | Milestone | Deliverable |
|---|---|---|---|
| Week 1 | 1–2 | Data Exploration | EDA notebook: distributions, passage lengths, question types, answer balance. Summary statistics table. |
| Week 1 | 3–4 | Preprocessing | `preprocessing.py` complete. Train/dev/test splits ready. One-Hot feature matrices saved. |
| Week 1 | 4–5 | Model A — Traditional ML | LR and SVM trained on One-Hot features. Accuracy + F1 reported on dev set. |
| Week 2 | 1–2 | Model A — Unsupervised/Semi-Supervised | Label Propagation or K-Means implemented. Purity/Silhouette score reported. Comparison table vs. supervised ML. |
| Week 2 | 3 | Model A — Ensemble | Soft-vote or stacking ensemble implemented. Final Model A test set results. |
| Week 2 | 3 | Model B — Traditional ML | One-Hot + cosine similarity distractor pipeline working. Word2Vec nearest-neighbour distractors evaluated (Precision, Recall, F1). |
| Week 2 | 4 | Model B Complete | ML-ranked distractor pipeline complete. Extractive hint scorer implemented and evaluated. |
| Week 2 | 4 | UI Development | All four screens implemented. Models integrated. Error handling complete. |
| Week 2 | 5 | Evaluation & Tuning | Full evaluation on RACE test set. Hyperparameter sweep. Human evaluation form completed. |
| Week 2 | 5 | Final Submission | Final report (PDF). GitHub repo with README. 10-min demo video or live presentation. |

---

## 14. Grading Rubric

| Component | Marks | Key Criteria |
|---|---|---|
| EDA & Preprocessing | 10 / 100 | Insightful visualizations; clean, documented pipeline |
| Model A — Traditional ML | 15 / 100 | ≥ 2 models, feature engineering, metric comparison table |
| Model A — Unsupervised / Semi-Supervised | 20 / 100 | At least one approach implemented and evaluated |
| Model A — Ensemble | 05 / 100 | Demonstrates improvement over individual models |
| Model B — Distractor Generation | 15 / 100 | Plausible distractors; Precision, Recall, F1, Confusion Matrix reported |
| Model B — Hint Generation | 10 / 100 | Graduated hints that guide without revealing answer |
| User Interface | 15 / 100 | All 4 screens present; smooth UX; error handling |
| Final Report | 05 / 100 | Clear methodology, results, discussion, limitations |
| Code Quality | 05 / 100 | Readable, documented code; meaningful commit history |
| **TOTAL** | **100 / 100** | |

---

## 15. Constraints & Ethical Considerations

### 15.1 Technical Constraints

- All models must be trainable on an NVIDIA RTX 3060 (12 GB) or Google Colab T4. Avoid architectures requiring > 24 GB VRAM.
- Inference for a single article + question must complete in **under 10 seconds** on target hardware.
- The full training pipeline must be reproducible from a single shell command or Jupyter notebook.
- Keep sparse matrices as sparse (use `scipy.sparse`). Do not call `.toarray()` on full RACE-sized matrices — this causes memory errors.

### 15.2 Ethical Considerations

Address all of the following in the final report:

- **Bias:** RACE passages come from Chinese school examinations. Discuss whether cultural or linguistic bias in the dataset affects model generalization to other populations or age groups.
- **Accessibility:** The UI must be usable by students with visual impairments — sufficient contrast ratios (WCAG AA) and full keyboard navigation.
- **Academic integrity:** Generated questions must not be deployed in real exam settings without human expert review.
- **Model transparency:** The UI must clearly indicate which answers and distractors are AI-generated and that errors are possible.

---

## 16. Deliverables Checklist

```
[ ] GitHub repository (shared with instructor) — clean commit history
[ ] requirements.txt with pinned versions
[ ] README.md (this file) with setup, training, and run instructions
[ ] EDA notebook (notebooks/EDA.ipynb) — all outputs visible
[ ] experiments.ipynb — hyperparameter search logs
[ ] Trained model checkpoints (models/ directory)
[ ] Final report PDF (proper documentation in paper/academic style)
[ ] 10-minute live demo session
[ ] Human evaluation forms (Rubric form)
```

### 16.1 Final Report Structure

1. Abstract (200 words max)
2. Introduction & Motivation
3. Related Work (cite at least 5 papers)
4. Dataset Analysis
5. Model A: Design, Training, Results
6. Model B: Design, Training, Results
7. User Interface Description
8. Evaluation & Discussion
9. Limitations & Future Work
10. Conclusion
11. References

---
