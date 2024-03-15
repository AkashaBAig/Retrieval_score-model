
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import openai
from dotenv import load_dotenv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the environment variables
load_dotenv()

# Set the OpenAI API key from the environment variables
openai.api_key = os.getenv('OPENAI_API')

class Selected_Model:
    def __init__(self, model_engine):
        self.model_engine = model_engine

    def get_response(self, retrieved, contexts, max_tokens, temp):
        retrieved = f"{retrieved}\nContexts:\n{contexts}"
        response = openai.Completion.create(
            model=self.model_engine,
            prompt=retrieved,
            max_tokens=max_tokens,
            temperature=temp
        )
        return response.choices[0].text.strip()

# Function to load dataset from CSV file using pandas
def load_csv_data(filename):
    df = pd.read_csv(filename, header=0)
    df.columns = ['retrieved', 'source']
    return df

# Function to compute similarity between the generated response and source
def compute_similarity(retrieved, source):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([retrieved, source])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return score

# Function to evaluate response using the similarity score
def evaluate_response(retrieved, source):
    similarity_score = compute_similarity(retrieved, source)
    score = int((similarity_score * 4) + 1)  # Scale the score to 1-5
    reasoning = "Based on computed similarity."
    return score, reasoning

# Function to evaluate responses using different parameters
def evaluate_responses(experiment_name, evaluator, temperature, max_tokens, system_content, user_content, experiments_dir, csv_filename, num_samples=None):
    references = load_csv_data(csv_filename)
    results = []
    selected_model = Selected_Model(evaluator)

    for index, row in tqdm(references.iterrows(), total=references.shape[0]):
        source = selected_model.get_response(row['retrieved'], system_content, max_tokens, temperature)
        score, reasoning = evaluate_response(row['retrieved'], source)
        result = {
            "Retrieved": row['retrieved'],
            "Source": source,
            "Score": score,
            "Reasoning": reasoning
        }
        results.append(result)

        # Print each result
        print(f"Result {index + 1}: {result}")

# Parameters for evaluating the response
experiment_name = "Sample Experiment"
evaluator = "text-davinci-003"
temperature = 0.7
max_tokens = 100
system_content = "You are a helpful assistant."
user_content = "Evaluate the following question and answer pairs."
experiments_dir = Path("path/to/experiments")
csv_filename = "Retrieval.csv"

# Execute the evaluation
evaluate_responses(experiment_name, evaluator, temperature, max_tokens, system_content, user_content, experiments_dir, csv_filename)
