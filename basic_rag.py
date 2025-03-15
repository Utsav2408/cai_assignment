import gdown
import os
import re
import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from typing import List
from transformers import AutoTokenizer
import pickle

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

"""# Data Collection"""

# For Data Collection
drive_links = {
    "Tesla 10K-24.pdf": "1xooioQYrvksABYBXxeqXKvrMjLSnf9FH",
    "Tesla 10K-25.pdf": "1q-znfmbHx4UYpBu4CeskIPI6rpT1eKwo"
}

# Directory for PDFs
download_dir = "pdf_files"
os.makedirs(download_dir, exist_ok=True)

# Function to download PDFs
def download_pdfs(drive_links, download_dir):
    pdf_files = {}
    for file_name, file_id in drive_links.items():
        output_path = os.path.join(download_dir, file_name)
        if not os.path.exists(output_path):  # Avoid re-downloading
            print(f"Downloading {file_name} Report...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
        pdf_files[file_name] = output_path
    return pdf_files

"""# Chunking"""

# Token-Based Text Chunking
def chunk_text(text, max_tokens=150):
    """
    Split text into smaller, fact-focused chunks suitable for financial analysis.
    Uses both sentence and paragraph markers while respecting token length limits.
    """

    # Split text into 'natural' financial paragraphs using known patterns
    raw_chunks = re.split(r'\n{2,}|(?<=\.)\s+(?=[A-Z])|(?<=%)\s+(?=[A-Z])', text)
    raw_chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]  # Clean up empty lines

    final_chunks = []
    current_chunk = ""
    current_token_count = 0

    for chunk in raw_chunks:
        # Estimate token count for this raw chunk
        chunk_token_count = len(tokenizer.tokenize(chunk))

        # If adding this chunk exceeds the max token limit, finalize current chunk
        if current_token_count + chunk_token_count > max_tokens:
            if current_chunk:
                final_chunks.append(current_chunk.strip())
            current_chunk = chunk  # Start new chunk
            current_token_count = chunk_token_count
        else:
            # Safe to add to current chunk
            if current_chunk:
                current_chunk += " " + chunk
            else:
                current_chunk = chunk
            current_token_count += chunk_token_count

    # Add any remaining chunk
    if current_chunk:
        final_chunks.append(current_chunk.strip())

    print(f"Smart financial chunking completed. Total {len(final_chunks)} chunks.")
    return final_chunks

# === Load and chunk PDFs ===
def load_and_chunk_pdfs(pdf_paths: List[str], max_tokens=150) -> List[str]:
    all_text = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            chunks = chunk_text(doc.page_content, max_tokens=max_tokens)
            all_text.extend(chunks)
    return all_text

"""# Vectorization"""

# === Embedding and FAISS indexing ===
def create_faiss_index(chunks: List[str], index_path: str, embed_model_name='sentence-transformers/all-MiniLM-L6-v2'):
    embedder = SentenceTransformer(embed_model_name)
    embeddings = embedder.encode(chunks, convert_to_tensor=False, show_progress_bar=True)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Save index and chunks
    faiss.write_index(index, f"{index_path}.index")
    with open(f"{index_path}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"[+] Index and chunks saved to {index_path}.index and {index_path}_chunks.pkl")

"""# Guard Rails

## Input Guard Rails
"""

# For Input Guard Rail

# Reference texts for classification
finance_reference = """
Tesla's financial performance, revenue, net income, earnings per share (EPS), gross margin,
operating income, profit and loss, quarterly and annual reports, balance sheet,
assets and liabilities, cash flow statements, research and development (R&D) spending,
capital expenditures, financial forecasts, debt-to-equity ratio, return on investment (ROI),
Tesla’s stock price history, Tesla’s SEC filings, investor relations,
fundamental analysis of Tesla, production costs, vehicle delivery numbers,
supply chain expenses, government subsidies, financial risks, and Tesla’s business strategy.
"""

non_finance_reference = """
Celebrity gossip, relationships, dating rumors, pop culture, viral TikTok trends,
sports news, football scores, basketball highlights, reality TV shows, Netflix recommendations,
music charts, entertainment industry updates, movie reviews, social media influencers,
meme culture, political debates, election results, conspiracy theories,
astrology predictions, video game discussions, cryptocurrency memes,
fashion trends, travel destinations, food recipes, cooking tips, and daily horoscopes.
"""

# Harmful words blacklist
harmful_keywords = [
    "kill", "attack", "hate", "racist", "violence", "bomb",
    "insult", "harass", "hack", "terror", "scam", "fraud"
]

def validate_user_query(user_query, level="moderate"):
    """
    Validates user queries based on the moderation level.
    Levels:
      - lenient: Blocks only harmful content.
      - moderate: Blocks off-topic queries, allows some open-ended finance Qs.
      - strict: Allows only fact-based Tesla finance queries.
    """

    # Check for harmful content
    if any(word in user_query.lower() for word in harmful_keywords):
        return False

    # Check for financial relevance using embeddings
    query_embeddings = embedding_model.encode([user_query, finance_reference, non_finance_reference], convert_to_tensor=True)
    finance_score = util.pytorch_cos_sim(query_embeddings[0], query_embeddings[1]).item()
    non_finance_score = util.pytorch_cos_sim(query_embeddings[0], query_embeddings[2]).item()

    # Define thresholds based on the level
    thresholds = {
        "lenient": 0.4,
        "moderate": 0.5,
        "strict": 0.7
    }

    # Decision based on similarity scores
    if finance_score >= thresholds[level] and finance_score > non_finance_score:
        return True

    return False

"""## Output Guard Rails"""

# For Output Guard Rail

# Tesla Finance Reference for Fallback Fact-Checking
finance_reference = """
Tesla's revenue, net income, financial statements, balance sheet, assets, liabilities,
R&D spending, cash flow, capital expenditures, gross margin, operating margin, earnings per share (EPS),
Tesla’s stock performance, shareholder reports, quarterly earnings, annual reports, SEC filings,
Tesla’s debt and equity, financial forecasts, vehicle production numbers, delivery reports,
Tesla’s partnerships, supply chain expenses, profit and loss statements, institutional investors,
Tesla’s investments in AI, energy sector revenue, carbon credits, and regulatory credits revenue.
"""

# Harmful or misleading content patterns
harmful_patterns = [
    r"Tesla is a scam", r"Tesla is going bankrupt", r"Tesla is doomed",
    r"illegal activities", r"fraudulent practices", r"criminal charges",
    r"will destroy the market", r"manipulating investors"
]

def filter_rag_response(user_query, model_response, retrieved_docs=None):
    """
    Filters RAG-generated responses using fact-checking, speculation detection, and safety filtering.
    Uses retrieved documents if available; otherwise, falls back to a finance reference.
    """

    # Fact-Checking: Compare Response with Retrieved Docs (only if available)
    reference_text = " ".join(retrieved_docs) if retrieved_docs else finance_reference
    response_embedding = embedding_model.encode([model_response, reference_text], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(response_embedding[0], response_embedding[1]).item()

    if similarity_score < 0.3:
        return "FYI, The response may be inaccurate. Please verify with Tesla's official reports.\n\n" + model_response

    # Harmful/Misleading Content Filtering
    for pattern in harmful_patterns:
        if re.search(pattern, model_response, re.IGNORECASE):
            return "This response has been blocked due to misleading content."

    # Low-Confidence Handling
    if "I think" in model_response or "It seems" in model_response:
        return "The model's confidence in this response is low. Please verify with reliable sources.\n\n" + model_response

    return model_response

"""# Retrieving chunks & generating answers using SLM

## Load Model
"""

# Basic RAG
slm = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

"""## Confidence Scores"""

def calculate_confidence_score(distances):
    """
    Calculate a confidence score based on distances from Faiss search.
    Lower distance → higher confidence.

    Returns:
        confidence (float): 0 to 1 score.
        band (str): 'High', 'Medium', 'Low'
    """
    if len(distances) == 0:
        return 0.0, "Low"

    # Normalize distances to [0, 1] range (lower is better)
    max_dist = max(distances)
    min_dist = min(distances)
    norm_distances = [(d - min_dist) / (max_dist - min_dist + 1e-6) for d in distances]

    # Confidence as inverse of average normalized distance
    avg_norm_distance = sum(norm_distances) / len(norm_distances)
    confidence = 1.0 - avg_norm_distance

    # Band assignment
    if confidence >= 0.7:
        band = "High"
    elif confidence >= 0.5:
        band = "Medium"
    else:
        band = "Low"

    return round(confidence, 2), band

"""## Basic RAG Retrieval"""

# === Step 3: Query and Use Flan-UL2 Small for Answering ===
async def query_rag(index_path: str, query: str, top_k=5):
    # Load index and chunks
    index = faiss.read_index(f"{index_path}.index")
    with open(f"{index_path}_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    query_vec = embedding_model.encode([query], convert_to_tensor=False)

    # Search
    distances, indices = index.search(np.array(query_vec), top_k)
    confidence, confidence_band = calculate_confidence_score(distances[0])
    relevant_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(relevant_chunks)

    # Build prompt
    prompt = f"""
You are a financial analyst. Given the following context from Tesla's reports, provide an answer to the Question concisely and factually.

Context:
{context}

Question: {query}

Answer:
"""

    answer = slm(prompt, max_length=400, truncation=True)[0]['generated_text']
    final_answer = filter_rag_response(query, answer.strip(), [chunk for chunk in relevant_chunks])

    return {
        "answer": final_answer,
        "confidence_score": confidence,
        "confidence_band": confidence_band
    }

"""# Dry Run"""

async def basic_rag_response(query):
  moderation_level = ['lenient', 'moderate', 'strict']
  validation = validate_user_query(query, level=moderation_level[0])

  if validation:
    print("Valid Query")
    answer = await query_rag("tesla_faiss_index", query)
    return answer
  else:
    return "Query Not Relevant: Please ask about Tesla's financial data."

import asyncio

def basic_generate_financial_response_sync(query):
    return asyncio.run(basic_rag_response(query))
