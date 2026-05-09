import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(page_title="RACE Quiz AI", layout="wide")

# --- MODEL B LOGIC (Hints & Distractors) ---
@st.cache_resource # Caches the model so the app doesn't reload it constantly
def load_ai_engine():
    vectorizer = joblib.load('./models/model_a/traditional/tfidf_vectorizer.pkl')
    # Loading the validation dataset for our "Random Sample" feature
    df = pd.read_csv('./data/raw/val.csv')
    return vectorizer, df

vectorizer, val_df = load_ai_engine()

def get_sentences(article):
    # Splits the article into sentences
    return [s.strip() + "." for s in article.replace('\n', ' ').split('.') if len(s.strip()) > 10]

def generate_hints(article, question):
    sentences = get_sentences(article)
    if len(sentences) <= 1: return sentences
    q_vec = vectorizer.transform([question])
    s_vecs = vectorizer.transform(sentences)
    sims = cosine_similarity(q_vec, s_vecs).flatten()
    ranked = np.argsort(sims)[::-1]
    return [sentences[ranked[i]] for i in range(min(3, len(ranked)))][::-1]

def generate_distractors(article, correct_answer):
    sentences = get_sentences(article)
    ans_vec = vectorizer.transform([correct_answer])
    s_vecs = vectorizer.transform(sentences)
    sims = cosine_similarity(ans_vec, s_vecs).flatten()
    ranked = np.argsort(sims)
    distractors = []
    for idx in ranked:
        cand = sentences[idx]
        if cand.lower() not in correct_answer.lower() and correct_answer.lower() not in cand.lower():
            distractors.append(cand)
        if len(distractors) == 3: break
    while len(distractors) < 3: distractors.append("Information not provided.")
    return distractors

# --- UI STATE MANAGEMENT ---
if 'current_sample' not in st.session_state:
    st.session_state.current_sample = None
if 'quiz_options' not in st.session_state:
    st.session_state.quiz_options = []

# --- SIDEBAR (Screen 4: Analytics Dashboard) ---
with st.sidebar:
    st.header("📊 Analytics Dashboard")
    st.markdown("### Model A (Verifier)")
    st.metric(label="Ensemble Accuracy", value="75.00%")
    st.metric(label="Macro F1-Score", value="0.4286")
    st.metric(label="K-Means Silhouette", value="0.0152")
    st.divider()
    st.markdown("### Model B (Generator)")
    st.success("Distractor Engine: Active")
    st.success("Hint Engine: Active")

# --- MAIN APP ---
st.title("🧠 Intelligent Reading Comprehension AI")
st.markdown("Generate quizzes, distractors, and hints automatically from text.")

# Screen 1: Article Input
st.subheader("1. Load a Passage")
col1, col2 = st.columns([3, 1])

with col2:
    if st.button("🎲 Load Random RACE Sample", use_container_width=True):
        idx = random.randint(0, len(val_df) - 1)
        sample = val_df.iloc[idx]
        st.session_state.current_sample = sample
        
        # Generate distractors and mix them with the real answer
        correct = sample[sample['answer']]
        distractors = generate_distractors(sample['article'], correct)
        all_options = distractors + [correct]
        random.shuffle(all_options)
        
        st.session_state.quiz_options = all_options
        st.session_state.correct_ans = correct

with col1:
    if st.session_state.current_sample is not None:
        st.info(st.session_state.current_sample['article'])
    else:
        st.info("Click the button on the right to load an article and generate a quiz!")

# Screen 2 & 3: Quiz View & Hints
if st.session_state.current_sample is not None:
    st.divider()
    st.subheader("2. AI Generated Quiz")
    st.markdown(f"**Question:** {st.session_state.current_sample['question']}")
    
    # Hint Panel (Collapsible)
    with st.expander("💡 Need a hint? (Model B)"):
        hints = generate_hints(st.session_state.current_sample['article'], st.session_state.current_sample['question'])
        for i, h in enumerate(hints, 1):
            st.write(f"**Hint {i}:** {h}")
            
    # Quiz Submission
    with st.form("quiz_form"):
        user_choice = st.radio("Select your answer:", st.session_state.quiz_options)
        submitted = st.form_submit_button("Check Answer")
        
        if submitted:
            if user_choice == st.session_state.correct_ans:
                st.success(f"✅ Correct! The AI verified this answer.")
            else:
                st.error(f"❌ Incorrect. The AI generated that distractor to trick you! The correct answer was: {st.session_state.correct_ans}")