from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass


from .config import Config
from .embeddings import Embedder


@dataclass
class CatalogIndex:
    embeddings: np.ndarray # shape: (N, D), assumed L2-normalized
    df: pd.DataFrame # same row order as embeddings


    def search_by_query(self, query: str, embedder: Embedder, top_k: int = 10) -> pd.DataFrame:
        q = embedder.model.encode(
        [embedder.cfg.bge_query_prefix + query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        )[0]
        # cosine == dot because normalized
        sims = self.embeddings @ q
        idx = np.argsort(-sims)[:top_k]
        out = self.df.iloc[idx].copy()
        out["score"] = sims[idx]
        return out


    def search_similar_to_product(self, product_id: str, top_k: int = 10) -> pd.DataFrame:
    # find row
        row = self.df.index[self.df["product_id"] == product_id]
        if len(row) == 0:
            raise KeyError(f"product_id {product_id} not found")
        i = int(row[0])
        v = self.embeddings[i]
        sims = self.embeddings @ v
        idx = np.argsort(-sims)[: top_k + 1]
        idx = idx[idx != i][:top_k] # drop self
        out = self.df.iloc[idx].copy()
        out["score"] = sims[idx]
        return out


    def search_by_recent(self, recent_ids: list[str], top_k: int = 10) -> pd.DataFrame:
        mask = self.df["product_id"].isin(recent_ids)
        if mask.sum() == 0:
            raise ValueError("None of the provided recent_ids were found in the catalog.")
        V = self.embeddings[mask.values]
        profile = V.mean(axis=0)
        # re-normalize the profile
        profile = profile / (np.linalg.norm(profile) + 1e-12)
        sims = self.embeddings @ profile
        # exclude already-viewed
        exclude = set(self.df.loc[mask, "product_id"].tolist())
        idx = np.argsort(-sims)
        picked = []
        for j in idx:
            pid = self.df.iloc[j]["product_id"]
            if pid in exclude:
                continue
            picked.append(j)
            if len(picked) == top_k:
                break
        out = self.df.iloc[picked].copy()
        out["score"] = sims[picked]
        return out