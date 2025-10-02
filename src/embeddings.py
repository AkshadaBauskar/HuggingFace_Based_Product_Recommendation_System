from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Iterable


from .config import Config


class Embedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name, device=cfg.device)


    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        embs = self.model.encode(
        list(texts),
        batch_size=self.cfg.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True, # L2 norms -> cosine == dot
        )
        return embs


    def embed_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        if "combined" not in df.columns:
            raise ValueError("Missing 'combined' column. Call build_combined_text first.")
        return self.encode_texts(df["combined"].astype(str).tolist())


    def save(self, embeddings: np.ndarray, df: pd.DataFrame):
        os.makedirs(os.path.dirname(self.cfg.embeddings_path), exist_ok=True)
        np.save(self.cfg.embeddings_path, embeddings)
    # Save id map to preserve row alignment on reload
        df[["product_id"]].reset_index(drop=False).rename(columns={"index": "row_idx"}).to_parquet(self.cfg.id_map_path, index=False)


    def load(self) -> tuple[np.ndarray, pd.DataFrame]:
        embs = np.load(self.cfg.embeddings_path)
        idmap = pd.read_parquet(self.cfg.id_map_path)
        return embs, idmap