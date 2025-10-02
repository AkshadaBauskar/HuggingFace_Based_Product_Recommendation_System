# Hugging Face–Based Product Recommendation System  

## Streamlit app: https://huggingfacebasedappuctrecommendationsystem-o7vryn96ceebtxwknvl.streamlit.app/

## Project Overview  
This project implements a semantic product recommendation system powered by **Hugging Face embeddings**. Instead of relying only on keywords, it understands the meaning of product titles and descriptions to suggest relevant items. The system supports three core use cases:  
1. **Free-text search** – Users can describe what they want in plain language.  
2. **Similar product recommendations** – Suggests items similar to a given product ID.  
3. **Browsing history personalization** – Generates recommendations based on recently viewed products.  
The architecture is modular, efficient, and production-ready, with both a Streamlit web app for interactive use and a CLI interface for testing and automation.  

## Problem Statement  
Traditional product recommenders often rely on keyword search or simple co-purchase data. These methods struggle with:  
- Capturing semantic similarity between products.  
- Handling user queries expressed in natural language.  
- Providing personalized recommendations from browsing history.  
This project addresses these limitations by leveraging **sentence-transformer embeddings** to represent products in vector space, making semantic similarity measurable and scalable.  

## Workflow & Components  

### 1. Catalog Embedding (`embed_catalog.py`)  
- Loads the dataset (`data/products_dataset.csv`) and combines product **title + description** into a single semantic field.  
- Generates embeddings using **Hugging Face BAAI/bge-small-en-v1.5** (384-dimensional vectors).  
- Saves artifacts in `artifacts/`:  
  - `embeddings.npy` → product embeddings.  
  - `id_map.parquet` → mapping between product IDs and embedding rows.  

### 2. Core Recommender (`recommender.py`)  
- Implements the **CatalogIndex** class to perform:  
  - `search_by_query()` – semantic free-text search.  
  - `search_similar_to_product()` – find similar items by product ID.  
  - `search_by_recent()` – recommendations based on recently viewed products.  
- Uses cosine similarity (dot product on L2-normalized embeddings).  

### 3. Data Utilities (`data_utils.py`)  
- `load_catalog()` – loads and validates the dataset.  
- `mark_recently_viewed()` – tracks product interactions for personalization.  
- `build_combined_text()` – prepares concatenated text for embeddings.  

### 4. Embeddings Manager (`embeddings.py`)  
- Wraps Hugging Face **SentenceTransformer** models.  
- Provides functions to encode text, embed DataFrames, and save/load artifacts.  

### 5. Configuration (`config.py`)  
Centralizes configuration (model name, device, batch size, file paths).  
- Default model: **BAAI/bge-small-en-v1.5**.  
- Auto-detects **GPU (CUDA)** if available.  

### 6. Interfaces  
- **Streamlit App (`app.py`)** – clean interactive dashboard with product search, similarity, and history-based recommendations.  
- **CLI Tool (`recommend_cli.py`)** – test recommendations via terminal using arguments like:  
  ```bash
  python recommend_cli.py --query "wireless headphones"
  python recommend_cli.py --similar_to 101
  python recommend_cli.py --recent_ids 23,56,77## 📊 Results & Capabilities  

- **Semantic Search**: Matches queries like *“budget wireless headphones”* with affordable headset products, even if exact keywords differ.  
- **Similar Products**: Finds alternatives for a given product without repeating the same item.  
- **Personalized History-Based Recommendations**: Learns user intent from recent interactions and suggests fresh, relevant items.  
- **Efficiency**: Precomputed embeddings allow instant responses without recomputation.  

## Tech Stack  

- **Programming:** Python  
- **Frameworks:** Hugging Face Transformers, SentenceTransformers, Streamlit  
- **Libraries:** Pandas, NumPy, Scikit-learn, PyTorch, argparse  
- **Techniques:** Semantic Embeddings, Cosine Similarity Search, Personalized Recommendations, Vector Indexing  

## 🚀 How to Run  

Clone the repository:  
```bash
git clone https://github.com/AkshadaBauskar/HuggingFace_Based_Product_Recommendation_System.git
cd HuggingFace_Based_Product_Recommendation_System

