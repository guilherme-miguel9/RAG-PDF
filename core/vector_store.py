import chromadb
from sentence_transformers import SentenceTransformer
from config import settings

# Inicializa o cliente do ChromaDB com persistência no diretório estruturado
client = chromadb.PersistentClient(path=settings.DB_PATH)

# Cache em memória para o modelo de embeddings (Bi-Encoder)
_embedding_model_cache = None


def get_embedding_model():
    """Retorna a instância do modelo Bi-Encoder (carregada em cache na memória)."""
    global _embedding_model_cache
    if _embedding_model_cache is None:
        print(f"🧠 [VectorStore] Carregando modelo Bi-Encoder: {settings.EMBEDDING_MODEL}...")
        _embedding_model_cache = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embedding_model_cache


def get_collection():
    """Retorna ou cria a coleção principal no ChromaDB com métrica de similaridade cosseno."""
    return client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def delete_collection():
    """Remove a coleção do ChromaDB caso seja solicitada reindexação forçada."""
    try:
        client.delete_collection(settings.COLLECTION_NAME)
        print(f"🗑️ [VectorStore] Coleção '{settings.COLLECTION_NAME}' removida com sucesso.")
    except Exception:
        pass


def get_collection_stats() -> dict:
    """Retorna estatísticas operacionais da coleção (total de chunks armazenados)."""
    try:
        coll = client.get_collection(settings.COLLECTION_NAME)
        count = coll.count()
        return {
            "exists": True,
            "count": count,
            "collection_name": settings.COLLECTION_NAME,
            "db_path": settings.DB_PATH
        }
    except Exception:
        return {
            "exists": False,
            "count": 0,
            "collection_name": settings.COLLECTION_NAME,
            "db_path": settings.DB_PATH
        }
