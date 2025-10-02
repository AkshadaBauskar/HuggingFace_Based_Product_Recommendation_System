from dataclasses import dataclass
import torch


@dataclass
class Config:
    model_name: str = "BAAI/bge-small-en-v1.5" # 384-dim, strong + fast
    batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Recommended BGE query prefix
    bge_query_prefix: str = "Represent this sentence for searching relevant passages: "


    # Files
    data_csv: str = "data/products_dataset.csv"
    embeddings_path: str = "artifacts/embeddings.npy"
    id_map_path: str = "artifacts/id_map.parquet"
    pca_2d_path: str = "artifacts/pca_2d.npy"