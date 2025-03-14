import gdown
import os
import fitz  # PyMuPDF
import camelot
from transformers import pipeline
from transformers import AutoTokenizer
import re
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util

# For Data Collection
drive_links = {
    "Tesla 10K-24.pdf": "1xooioQYrvksABYBXxeqXKvrMjLSnf9FH",
    "Tesla 10K-25.pdf": "1q-znfmbHx4UYpBu4CeskIPI6rpT1eKwo"
}

# Directory for PDFs
download_dir = "pdf_files"
os.makedirs(download_dir, exist_ok=True)

# For Data Extraction
TEXT_OUTPUT_DIR = "extracted_text"
TABLES_OUTPUT_DIR = "extracted_tables_natural"

os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLES_OUTPUT_DIR, exist_ok=True)

slm_summarizer = pipeline("text2text-generation", model="google/flan-t5-small")

# List of your Tesla 10-K files
pdf_files = [
    "pdf_files/Tesla 10K-24.pdf",
    "pdf_files/Tesla 10K-25.pdf"
]

# For Chunking
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# For Faiss
FAISS_DIR = "faiss_indices"
os.makedirs(FAISS_DIR, exist_ok=True)

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

# Adv RAG

slm = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

"""# Data Collection"""

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

"""# Data Extraction"""

def extract_text_from_pdf(pdf_path: str, output_filename: str):
    print(f"Extracting text from {pdf_path}...")
    doc = fitz.open(pdf_path)
    all_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        all_text += page.get_text() + "\n"
    output_path = os.path.join(TEXT_OUTPUT_DIR, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(all_text)
    print(f"Text saved to: {output_path}")

def extract_and_summarize_tables(pdf_path: str, output_filename: str):
    print(f"Extracting tables from {pdf_path}...")
    tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
    summarized_tables = ""

    for i, table in enumerate(tables):
        try:
            table_str = table.df.to_string(index=False)
            print(f"\nTable {i+1} extracted. Converting to natural language summary...")

            # Generate a summary using SLM
            prompt = f"""You are a financial analyst. Given the following financial table, summarize the key insights accurately without altering any numbers.
    Focus on comparing figures across periods, and highlight trends, growth, or decline.
    Be precise and concise. Do not add information that is not present in the table. Here's the table:

    {table_str}

    Provide a brief summary suitable for a business report."""
            summary = slm_summarizer(prompt, max_length=300, truncation=True)[0]['generated_text']
            summarized_tables += f"Table {i+1} Summary:\n{summary}\n\n"

        except Exception as e:
            print(f"Error processing table {i+1}: {e}")

    output_path = os.path.join(TABLES_OUTPUT_DIR, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summarized_tables)
    print(f"Table summaries saved to: {output_path}")

def process_pdf(file_path: str):
    file_prefix = os.path.splitext(os.path.basename(file_path))[0].replace(" ", "_")
    extract_text_from_pdf(file_path, f"{file_prefix}_text.txt")
    extract_and_summarize_tables(file_path, f"{file_prefix}_tables.txt")

"""# Chunking (Chunk Merging)"""

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


# Function to load and chunk text
def load_and_chunk_text(file_path, max_tokens=150):
    """
    Load text file and apply 'matter-of-fact' chunking for financial analysis.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Apply refined chunking
    chunks = chunk_text(text, max_tokens=max_tokens)
    return chunks

# Function to load table summaries
def load_table_summaries(file_path):
    """Loads table summaries and splits based on 'Table X Summary:' markers."""
    with open(file_path, 'r', encoding='utf-8') as f:
        tables = f.read()
    table_chunks = re.split(r'Table \d+ Summary:', tables)
    table_chunks = [f"Table {i+1} Summary: {chunk.strip()}" for i, chunk in enumerate(table_chunks) if chunk.strip()]
    print(f"Loaded {len(table_chunks)} table summaries.")
    return table_chunks

# Semantic Matching for Tables to Text
def semantic_match_tables_to_text(text_chunks, table_chunks, text_vectors, table_vectors):
    """Matches each table summary to the closest text chunk using semantic similarity."""
    matches = []
    for idx, table_vec in enumerate(table_vectors):
        similarities = np.dot(text_vectors, table_vec)
        best_match_idx = np.argmax(similarities)  # Get the most relevant text chunk
        matches.append((idx, best_match_idx, similarities[best_match_idx]))  # (table index, matched text index, score)

    print(f"Matched {len(matches)} tables to text sections.")
    return matches

# Merging Text and Table Chunks
def merge_based_on_matching(text_chunks, table_chunks, matches):
    """
    Merges each table summary into its most relevant text chunk based on semantic similarity.
    Also returns unmatched tables for separate storage.
    """
    merged_document = ""
    added_tables = set()

    for idx, text_chunk in enumerate(text_chunks):
        merged_document += f"\n\n### Section {idx+1}\n{text_chunk}\n"
        # Find and insert matching table
        for table_idx, matched_idx, score in matches:
            if matched_idx == idx and table_idx not in added_tables:
                merged_document += "\nRelated Table Summary:\n"
                merged_document += table_chunks[table_idx] + "\n"
                added_tables.add(table_idx)

    # Collect unmatched tables
    unmatched_tables = []
    for idx, table_chunk in enumerate(table_chunks):
        if idx not in added_tables:
            merged_document += f"\n\nUnmatched Table Summary:\n{table_chunk}\n"
            unmatched_tables.append((idx + 1, table_chunk))  # Store table index (1-based) and content

    return merged_document, unmatched_tables

"""# Vectorization"""

# Vectorization
def vectorize_chunks(chunks):
    embeddings = embedding_model.encode(chunks, convert_to_tensor=False, normalize_embeddings=True)
    return np.array(embeddings).astype('float32')

# Vector Preparation for Faiss
def prepare_vectors_with_metadata(source_label, text_chunks, table_chunks):
    metadata = []
    combined_chunks = []

    for idx, chunk in enumerate(text_chunks):
        metadata.append({"chunk": chunk, "source": source_label, "type": "text", "section": idx + 1})
        combined_chunks.append(chunk)

    for idx, chunk in enumerate(table_chunks):
        metadata.append({"chunk": chunk, "source": source_label, "type": "table", "section": idx + 1})
        combined_chunks.append(chunk)

    vectors = vectorize_chunks(combined_chunks)
    return vectors, metadata

# Faiss Index Builder
def build_faiss(sources_info, merged_dir, index_file, metadata_file):
    """
    Build Faiss index from final merged chunks, including unmatched tables as standalone chunks.
    """
    all_vectors = []
    all_metadata = []

    for source_label, merged_file_name in sources_info:
        # Load merged file
        merged_path = os.path.join(merged_dir, merged_file_name)
        with open(merged_path, 'r', encoding='utf-8') as f:
            merged_text = f.read()

        # Split merged content into sections (assuming '### Section X' markers used)
        merged_chunks = re.split(r'### Section \d+\n', merged_text)
        merged_chunks = [chunk.strip() for chunk in merged_chunks if chunk.strip()]

        print(f"Loaded {len(merged_chunks)} merged chunks from {merged_file_name}")

        # Prepare and vectorize merged chunks
        vectors = embedding_model.encode(merged_chunks, convert_to_tensor=False, normalize_embeddings=True)
        vectors = np.array(vectors).astype('float32')
        all_vectors.append(vectors)

        # Add metadata for merged chunks
        for idx, chunk in enumerate(merged_chunks):
            meta = {
                "chunk": chunk,
                "source": source_label,
                "type": "merged",  # Marked as merged content (text + tables)
                "section": idx + 1
            }
            all_metadata.append(meta)

        # Extract unmatched tables directly from merged file (based on our consistent formatting)
        unmatched_tables = re.findall(r'Unmatched Table Summary:\n(.*?)(?=\n\n|$)', merged_text, re.DOTALL)
        print(f"Found {len(unmatched_tables)} unmatched tables in {merged_file_name}")

        if unmatched_tables:
            # Vectorize unmatched tables
            unmatched_vectors = embedding_model.encode(unmatched_tables, convert_to_tensor=False, normalize_embeddings=True)
            unmatched_vectors = np.array(unmatched_vectors).astype('float32')
            all_vectors.append(unmatched_vectors)

            # Metadata for unmatched tables
            for idx, chunk in enumerate(unmatched_tables):
                meta = {
                    "chunk": chunk.strip(),
                    "source": source_label,
                    "type": "table",  # Clearly marked as standalone table
                    "section": f"unmatched-{idx + 1}"
                }
                all_metadata.append(meta)

    # Combine all vectors
    final_vectors = np.vstack(all_vectors)
    print(f"Total combined vectors (merged + unmatched tables): {len(final_vectors)}")

    # Build Faiss index
    dimension = final_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(final_vectors)
    faiss.write_index(index, index_file)
    print(f"Final Faiss index saved: {index_file}")

    # Save metadata
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2)
    print(f"Metadata including unmatched tables saved: {metadata_file}")

# Adaptive Search Function
async def adaptive_search(index_file, metadata_file, query, top_k=5):
    """
    Search the Faiss index and return top_k relevant chunks with metadata and distance scores.
    """
    index = faiss.read_index(index_file)
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    query_vector = embedding_model.encode([query], convert_to_tensor=False, normalize_embeddings=True).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        data = metadata[idx]
        data['score'] = dist  # Raw distance score
        results.append(data)
    return results, distances[0]  # Return raw distances for confidence calculation

"""# Run Chunking & Vectorization"""

def full_pipeline(text_file, table_file, output_file, source_label, merged_dir="merged"):
    """
    Complete pipeline to chunk, match, merge, and save merged content.
    Now handles merged content and unmatched tables separately.
    """

    print(f"\n Running pipeline for {source_label}")

    # Chunk text and tables
    text_chunks = load_and_chunk_text(text_file)
    table_chunks = load_table_summaries(table_file)

    # Vectorize chunks for semantic matching
    text_vectors = vectorize_chunks(text_chunks)
    table_vectors = vectorize_chunks(table_chunks)

    # Match tables to relevant text sections
    matches = semantic_match_tables_to_text(text_chunks, table_chunks, text_vectors, table_vectors)

    # Merge text and tables based on semantic match — returns (merged_doc, unmatched_tables)
    merged_doc, unmatched_tables = merge_based_on_matching(text_chunks, table_chunks, matches)

    # Save merged document to disk
    os.makedirs(merged_dir, exist_ok=True)
    merged_output_path = os.path.join(merged_dir, output_file)
    with open(merged_output_path, 'w', encoding='utf-8') as f:
        f.write(merged_doc)
    print(f"Merged document saved to: {merged_output_path}")

    # Return merged output file name and unmatched tables for indexing later
    return output_file, source_label, unmatched_tables

"""# Guard Rails

## Input Guard Rails
"""

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

"""# Advanced RAG Implementation

## Classify the query - Simple, Moderate, Complex
"""

# Function to classify query
async def classify_query_complexity(query):
    classifier_prompt = f"""
Classify this query as one of the following categories:
- Simple
- Moderate
- Complex

Query: {query}

Category:
"""
    result = slm(classifier_prompt, max_length=10)[0]['generated_text'].strip().lower()
    if "simple" in result:
        return "simple"
    elif "moderate" in result:
        return "moderate"
    else:
        return "complex"

"""## Generate Answer function for Simple & Moderate queries"""

# Answer generator from retrieved context
async def generate_answer_with_context(query, context, slm_pipeline):
    final_prompt = f"""
You are a financial analyst. Given the following context from Tesla's reports, provide an answer to the Question concisely and factually.

Context:
{context}

Question: {query}

Answer:
"""
    response = slm_pipeline(final_prompt, max_length=250, truncation=True)[0]['generated_text']
    return response

"""## Calculate confidence scores"""

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
    if confidence >= 0.5:
        band = "High"
    elif confidence >= 0.3:
        band = "Medium"
    else:
        band = "Low"

    return round(confidence, 2), band

"""## Adaptive Retrieval"""

async def adaptive_retrieve_and_answer(query, index_file, metadata_file, slm_pipeline, max_k=5):
    """
    Adaptive retrieval and answer generation, including confidence scoring.
    """

    # Classify query complexity
    query_type = await classify_query_complexity(query)
    print(f"🔍 Query classified as: {query_type}")

    # Perform adaptive search and get distances
    if query_type == "simple" or query_type == "moderate":
        results, distances = await adaptive_search(index_file, metadata_file, query, top_k=max_k)
    elif query_type == "complex":
        results, distances = await adaptive_search(index_file, metadata_file, query, top_k=max_k * 2)

    # Calculate confidence score
    confidence, confidence_band = calculate_confidence_score(distances)

    # Build context for SLM
    context = "\n".join([f"- {res['chunk']}" for res in results])
    # print(f"\n📜 Retrieved Context:\n{context}\n")

    # Generate answer based on query type
    if query_type == "simple" or query_type == "moderate":
        answer = await generate_answer_with_context(query, context, slm_pipeline)
    elif query_type == "complex":
        complex_prompt = f"""
You are a Tesla financial expert. Based on the following data, provide a detailed and reasoned answer to the question.

Context:
{context}

Question: {query}

Detailed Answer:
"""
        answer = slm_pipeline(complex_prompt, max_length=400, truncation=True)[0]['generated_text']

    # Apply output guard rail
    final_answer = filter_rag_response(query, answer.strip(), [res['chunk'] for res in results])

    # Return final answer with confidence
    return {
        "answer": final_answer,
        "confidence_score": confidence,
        "confidence_band": confidence_band
    }

async def generate_financial_response(query):
  # INPUT GUARD RAIL...
  moderation_level = ['lenient', 'moderate', 'strict']
  validation = validate_user_query(query, level=moderation_level[0])

  if validation:
    print("Valid Query")
    # Run adaptive retrieval with confidence
    result = await adaptive_retrieve_and_answer(
        query,
        "tesla_10K_unified_index.faiss",
        "tesla_10K_metadata.json",
        slm,
        max_k=5
    )

    return result
  else:
    return "Query Not Relevant: Please ask about Tesla's financial data."

import asyncio

def generate_financial_response_sync(query):
    return asyncio.run(generate_financial_response(query))
