import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import settings

_embeddings_cache = None
_vector_db_cache = None


def get_embeddings():
    """Retorna a instância em cache do modelo de embeddings HuggingFace (Bi-Encoder)."""
    global _embeddings_cache
    if _embeddings_cache is None:
        print(f"[LangChain VectorStore] Carregando modelo de embeddings: {settings.EMBEDDING_MODEL}")
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings_cache


def get_vector_db():
    """Retorna ou inicializa o banco vetorial Chroma via integração nativa do LangChain."""
    global _vector_db_cache
    if _vector_db_cache is None:
        embeddings = get_embeddings()
        _vector_db_cache = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=settings.DB_PATH,
            collection_metadata={"hnsw:space": "cosine"}
        )
    return _vector_db_cache


def delete_collection():
    """Limpa a base vetorial reinstanciando o repositório no ChromaDB via LangChain."""
    global _vector_db_cache
    try:
        if _vector_db_cache is not None:
            _vector_db_cache.delete_collection()
            _vector_db_cache = None
        else:
            db = get_vector_db()
            db.delete_collection()
            _vector_db_cache = None
        print(f"[LangChain VectorStore] Coleção '{settings.COLLECTION_NAME}' removida com sucesso.")
    except Exception as e:
        print(f"[LangChain VectorStore] Aviso ao remover coleção: {e}")


def get_collection_stats() -> dict:
    """Retorna estatísticas operacionais da base vetorial do LangChain."""
    try:
        db = get_vector_db()
        # O cliente nativo interno do Chroma no LangChain
        count = db._collection.count()
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
