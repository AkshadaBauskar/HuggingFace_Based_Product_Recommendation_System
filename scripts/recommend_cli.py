import argparse
import numpy as np
import pandas as pd


from src.config import Config
from src.data_utils import load_catalog
from src.embeddings import Embedder
from src.recommender import CatalogIndex


parser = argparse.ArgumentParser(description="HF recommender CLI")
parser.add_argument("--query", type=str, help="Free-text query", default=None)
parser.add_argument("--similar_to", type=str, help="Product ID to find similars", default=None)
parser.add_argument("--recent_ids", type=str, help="Comma-separated product_ids", default=None)
parser.add_argument("--top_k", type=int, default=10)
args = parser.parse_args()


cfg = Config()
print("Loading catalog & artifacts…")
# catalog to keep metadata aligned with embeddings
df = load_catalog(cfg.data_csv)
emb = Embedder(cfg)
X, idmap = emb.load()


# align df to saved id order (row_idx)
order = idmap.sort_values("row_idx")["product_id"].tolist()
lookup = {pid: i for i, pid in enumerate(df["product_id"]) }
idxs = [lookup[pid] for pid in order]
df = df.iloc[idxs].reset_index(drop=True)


index = CatalogIndex(embeddings=X, df=df)


if args.query:
    print("Searching by query…")
    res = index.search_by_query(args.query, embedder=emb, top_k=args.top_k)
    print(res[["product_id","title","score"]].to_string(index=False))
elif args.similar_to:
    print("Searching products similar to:", args.similar_to)
    res = index.search_similar_to_product(args.similar_to, top_k=args.top_k)
    print(res[["product_id","title","score"]].to_string(index=False))
elif args.recent_ids:
    recent = [x.strip() for x in args.recent_ids.split(",") if x.strip()]
    print("Searching by recent IDs:", recent)
    res = index.search_by_recent(recent, top_k=args.top_k)
    print(res[["product_id","title","score"]].to_string(index=False))
else:
    parser.print_help()