import re
from sentence_transformers import CrossEncoder
from config import settings
from core import vector_store

# Cache em memória para o modelo Cross-Encoder (Reranker)
_reranker_model_cache = None


def get_reranker_model():
    """Carrega ou retorna a instância em cache do modelo de Reranking Cross-Encoder."""
    global _reranker_model_cache
    if _reranker_model_cache is None:
        print(f"🎯 [Retriever] Carregando modelo Cross-Encoder: {settings.RERANKER_MODEL}...")
        _reranker_model_cache = CrossEncoder(settings.RERANKER_MODEL)
    return _reranker_model_cache


def expand_query(query: str) -> str:
    """
    Expande a query do usuário adicionando termos correlatos para maximizar o recall no Estágio 1.
    """
    expansions = {
        r"list[ae]|liste|quais (são os|os|as)|enumere": "lista itens enumeração requisitos",
        r"o que [eé]|explique|defina|definição|significa": "definição conceito descrição explicação significado",
        r"como (fazer|realizar|executar|funciona|é feito|proceder)": "procedimento passo etapa modo instrução execução",
        r"procedimento|protocolo|processo|rotina": "procedimento protocolo etapas instruções passos rotina",
        r"nota[s]?|observa[cç][aã]o|aviso|atenção": "nota observação aviso atenção importante cuidado",
        r"objetivo[s]?|finalidade|para que": "objetivo finalidade propósito meta razão",
        r"responsável|quem (deve|faz|realiza)": "responsável executor função encarregado cargo",
    }
    expanded = query
    for pattern, extra in expansions.items():
        if re.search(pattern, query, re.IGNORECASE):
            expanded = f"{query} {extra}"
            break
    return expanded


def retrieve(
    query: str,
    n_retrieval: int = settings.TOP_K_RETRIEVAL,
    n_rerank: int = settings.TOP_K_RERANK,
    use_reranker: bool = True
) -> list[dict]:
    """
    Executa a recuperação em 2 estágios:
    1. Busca vetorial rápida por similaridade de cosseno (ChromaDB + Bi-Encoder).
    2. Refinamento e reordenação de precisão (Cross-Encoder Reranker).
    """
    collection = vector_store.get_collection()
    embed_model = vector_store.get_embedding_model()
    
    # 1. Expansão de Query para Estágio 1
    expanded_query = expand_query(query)
    
    # 2. Embedding da consulta
    query_text_for_embedding = expanded_query
    if "e5" in settings.EMBEDDING_MODEL.lower():
        query_text_for_embedding = f"query: {expanded_query}"
        
    query_vector = embed_model.encode([query_text_for_embedding], normalize_embeddings=True)[0]
    
    # 3. Consulta ao ChromaDB (Estágio 1)
    try:
        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=n_retrieval,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        print(f"❌ [Retriever] Erro ao consultar o banco vetorial: {e}")
        return []
        
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    if not docs:
        return []
        
    candidates = []
    for doc, meta, dist in zip(docs, metas, distances):
        candidates.append({
            "text": doc,
            "page_num": meta.get("page_num", "N/A"),
            "source": meta.get("source", "PDF"),
            "vector_distance": round(dist, 4)
        })
        
    if not use_reranker or len(candidates) <= 1:
        return candidates[:n_rerank]
        
    # 4. Reranking com Cross-Encoder (Estágio 2)
    # A comparação do Cross-Encoder é sempre feita usando a pergunta original (limpa) do usuário
    reranker = get_reranker_model()
    pairs = [(query, c["text"]) for c in candidates]
    
    try:
        scores = reranker.predict(pairs)
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(round(score, 4))
            
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    except Exception as e:
        print(f"⚠️ [Retriever] Aviso no Reranker ({e}). Usando ordenação original do Bi-Encoder.")
        candidates.sort(key=lambda x: x["vector_distance"])
        
    return candidates[:n_rerank]