import re
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from config import settings
from core import vector_store

_reranker_model_cache = None


def get_reranker_model():
    """Carrega em cache o modelo Cross-Encoder (Reranker) de alta precisão."""
    global _reranker_model_cache
    if _reranker_model_cache is None:
        print(f"[LangChain Retriever] Carregando modelo Cross-Encoder Reranker: {settings.RERANKER_MODEL}")
        _reranker_model_cache = CrossEncoder(settings.RERANKER_MODEL)
    return _reranker_model_cache


def retrieve(query: str, top_k_retrieval: int = settings.TOP_K_RETRIEVAL, top_k_rerank: int = settings.TOP_K_RERANK) -> list[dict]:
    """
    Executa a recuperação em 2 estágios utilizando a estrutura LangChain:
    1. Retrieval com Bi-Encoder via Chroma (Busca de similaridade com pontuação no LangChain).
    2. Reranking com Cross-Encoder de todos os candidatos.
    Retorna uma lista de dicionários padronizada com conteúdo, pontuações e metadados.
    """
    db = vector_store.get_vector_db()
    
    # 1. Estágio 1: Retrieval Inicial no LangChain
    # O Chroma no LangChain retorna tuplas (Document, score) onde score na métrica cosseno varia segundo a distância
    docs_with_scores = db.similarity_search_with_score(query, k=top_k_retrieval)
    
    if not docs_with_scores:
        print(f"[LangChain Retriever] Nenhum documento recuperado para a busca: '{query}'")
        return []
        
    candidates = []
    for doc, dist_score in docs_with_scores:
        # Em cosseno no Chroma: similaridade = 1.0 - dist_score
        sim_score = max(0.0, float(1.0 - dist_score))
        candidates.append({
            "text": doc.page_content,
            "page_num": doc.metadata.get("page_num", 1),
            "similarity": sim_score,
            "langchain_doc": doc
        })
        
    # Filtro de relevância semântica mínima
    threshold = getattr(settings, "SIMILARITY_THRESHOLD", 0.15)
    candidates = [c for c in candidates if c["similarity"] >= threshold]
    
    if not candidates:
        print(f"[LangChain Retriever] Documentos abaixo do limiar de similaridade ({threshold}).")
        return []
        
    # 2. Estágio 2: Reranking com Cross-Encoder
    reranker = get_reranker_model()
    pairs = [[query, c["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)
    
    for i, score in enumerate(rerank_scores):
        candidates[i]["rerank_score"] = float(score)
        
    # Ordena decrescente pelo score do Cross-Encoder
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    return candidates[:top_k_rerank]


def clean_query(query: str) -> str:
    """Normaliza a consulta removendo caracteres de controle."""
    query = re.sub(r"\s+", " ", query)
    return query.strip()