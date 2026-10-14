"""Configuración central del proyecto.

Toda credencial y URL vive en .env (fuera del código y del git).
Cualquier script del proyecto importa de aquí:
    from config import EMBEDDINGS, LLM, PG
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# --- Embeddings (NAN: qwen3-embedding) ---
EMBEDDINGS = {
    "base_url": os.environ["NAN_BASE_URL"],
    "api_key": os.environ["NAN_API_KEY"],
    "model": "qwen3-embedding",
    # Truncado matryoshka: pedimos 1024 en vez de los 4096 nativos para que
    # pgvector pueda indexar el vector con HNSW (límite: 2000 dimensiones).
    "dim": int(os.environ.get("EMBEDDINGS_DIM", "1024")),
}

# --- Rerank (NAN: Qwen3-Reranker, se usará en la Fase 6) ---
RERANK = {
    "base_url": os.environ["NAN_BASE_URL"],
    "api_key": os.environ["NAN_API_KEY"],
    "model": "rerank",
}

# --- LLM (proxy LiteLLM propio) ---
LLM = {
    "base_url": os.environ["LITELLM_BASE_URL"],
    "api_key": os.environ["LITELLM_API_KEY"],
    "model": "nan/glm5.3-flash",
}

# --- Vector store (Postgres + pgvector en el nuc) ---
PG = {
    "host": os.environ["PG_HOST"],
    "port": int(os.environ["PG_PORT"]),
    "dbname": os.environ["PG_DB"],
    "user": os.environ["PG_USER"],
    "password": os.environ["PG_PASSWORD"],
}
