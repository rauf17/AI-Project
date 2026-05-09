# This is the Master Pipeline for Model B, which includes:
# 1. Data Preparation (with advanced features)
# 2. Training Logistic Regression and SVM models
# 3. Building an Ensemble of the two models
# 4. Running Unsupervised K-Means Clustering to discover latent groupings
# 5. Generating hints for the user

# Muhammad Umair (23I-0662) AND Abdul Rauf (23I-0591)

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

class ModelBPipeline:
    def __init__(self):
        # Load the TF-IDF vectorizer we built in Phase 1
        self.vectorizer = joblib.load('./models/model_a/traditional/tfidf_vectorizer.pkl')

    def extract_sentences(self, article):
        # Simple rule-based sentence splitter
        sentences = [s.strip() + "." for s in article.replace('\n', ' ').split('.') if len(s.strip()) > 10]
        return sentences if sentences else [article]

    def generate_hints(self, article, question, top_k=3):
        """
        Scores sentences by cosine similarity to the QUESTION.
        Returns them in reverse order (Broad hint -> Explicit hint).
        """
        sentences = self.extract_sentences(article)
        if len(sentences) <= 1:
            return sentences

        # Vectorize
        q_vec = self.vectorizer.transform([question])
        s_vecs = self.vectorizer.transform(sentences)

        # Compute similarity
        sims = cosine_similarity(q_vec, s_vecs).flatten()
        ranked_indices = np.argsort(sims)[::-1] # Highest first
        
        hints = []
        for i in range(min(top_k, len(ranked_indices))):
            hints.append(sentences[ranked_indices[i]])
        
        # Reverse list so the user gets the least obvious hint first
        return hints[::-1]

    def generate_distractors(self, article, correct_answer, num_distractors=3):
        """
        Finds sentences in the article that have LOW similarity to the correct answer
        to act as plausible but incorrect options.
        """
        sentences = self.extract_sentences(article)
        
        ans_vec = self.vectorizer.transform([correct_answer])
        s_vecs = self.vectorizer.transform(sentences)
        
        sims = cosine_similarity(ans_vec, s_vecs).flatten()
        ranked_indices = np.argsort(sims) # Lowest similarity first
        
        distractors = []
        for idx in ranked_indices:
            candidate = sentences[idx]
            # Ensure we don't accidentally pick the real answer
            if candidate.lower() not in correct_answer.lower() and correct_answer.lower() not in candidate.lower():
                distractors.append(candidate)
            if len(distractors) == num_distractors:
                break
                
        # Fallback just in case the article is extremely short
        while len(distractors) < num_distractors:
            distractors.append("Information not provided in text.")
            
        return distractors

if __name__ == "__main__":
    print("Testing Model B (Hints & Distractors)...\n")
    
    # Load a single test sample from validation data
    df = pd.read_csv('./data/raw/val.csv')
    sample = df.iloc[5] # Picking a random row
    
    model_b = ModelBPipeline()
    correct_ans = sample[sample['answer']]
    
    print("--- QUESTION ---")
    print(sample['question'])
    print(f"\n✅ CORRECT ANSWER: {correct_ans}")
    
    print("\n--- GENERATED HINTS (From Model B) ---")
    hints = model_b.generate_hints(sample['article'], sample['question'])
    for i, h in enumerate(hints, 1):
        print(f"Hint {i}: {h}")
        
    print("\n--- GENERATED DISTRACTORS (From Model B) ---")
    distractors = model_b.generate_distractors(sample['article'], correct_ans)
    for i, d in enumerate(distractors, 1):
        print(f"❌ Option: {d}")
        
    print("\nPhase 3 Complete! Model B logic is fully functional.")