import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # force Transformers to skip TF/Keras
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # optional: cleaner logs
import numpy as np
import pandas as pd
import streamlit as st
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data_utils import load_catalog
from src.embeddings import Embedder
from src.recommender import CatalogIndex

st.set_page_config(page_title="HF Product Recommender", layout="wide")
st.title("🛍️ Product Recommender (Hugging Face, Local Embeddings)")


cfg = Config()


@st.cache_resource
def load_index() -> CatalogIndex:
    df = load_catalog(cfg.data_csv)
    emb = Embedder(cfg)
    X, idmap = emb.load()
    # align df to saved id order
    order = idmap.sort_values("row_idx")["product_id"].tolist()
    lookup = {pid: i for i, pid in enumerate(df["product_id"]) }
    idxs = [lookup[pid] for pid in order]
    df = df.iloc[idxs].reset_index(drop=True)
    return CatalogIndex(embeddings=X, df=df), emb


index, emb = load_index()


col1, col2 = st.columns(2)
with col1:
    st.subheader("🔎 Search by query")
    query = st.text_input("Describe what you want", value="budget wireless headphones with noise isolation")
    top_k = st.slider("How many results?", 1, 30, 10)
    if st.button("Search"):
        res = index.search_by_query(query, embedder=emb, top_k=top_k)
        st.dataframe(res[["product_id","title","score"]])


with col2:
    st.subheader("🧩 Similar to a product")
    pid = st.text_input("Product ID", value="")
    top_k_2 = st.slider("Results (similar)", 1, 30, 10, key="topk2")
    if st.button("Find similars") and pid:
        try:
            res = index.search_similar_to_product(pid, top_k=top_k_2)
            st.dataframe(res[["product_id","title","score"]])
        except KeyError as e:
            st.warning(str(e))


st.divider()
st.subheader("🕘 Recommend from recently viewed")
recent = st.text_input("Comma-separated product_ids", value="P1938,P1970,P1044")
if st.button("Recommend"):
    recent_ids = [x.strip() for x in recent.split(",") if x.strip()]
    try:
        res = index.search_by_recent(recent_ids, top_k=10)
        cols = [c for c in ["product_id","title","price","url","score"] if c in res.columns]
        st.dataframe(res[cols])
    except ValueError as e:
        st.warning(str(e))


st.caption("Model: BAAI/bge-small-en-v1.5 • Vectors L2-normalized • Cosine similarity = dot product")