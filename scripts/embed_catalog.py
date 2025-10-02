import numpy as np
import pandas as pd
from pathlib import Path


from src.config import Config
from src.data_utils import load_catalog, build_combined_text
from src.embeddings import Embedder


if __name__ == "__main__":
    cfg = Config()
    print("Loading catalog…")
    df = load_catalog(cfg.data_csv)
    df = build_combined_text(df)


    print(f"Loading model: {cfg.model_name}")
    emb = Embedder(cfg)
    print("Encoding texts…")
    X = emb.embed_dataframe(df)
    print("Embeddings shape:", X.shape)


    print("Saving artifacts…")
    emb.save(X, df)
    print("Done. Artifacts written to:", cfg.embeddings_path)