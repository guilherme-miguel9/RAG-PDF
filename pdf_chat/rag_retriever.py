import re
from sentence_transformers import CrossEncoder
import config
from rag_indexer import get_collection, get_embedding_model

# Cache em memória do modelo Cross-Encoder (Reranker)
_reranker_cache = None


def get_reranker_model():
    """Carrega o modelo de Reranking Cross-Encoder (com cache em memória)."""
    global _reranker_cache
    if _reranker_cache is None:
        print(f"🎯 Carregando modelo Cross-Encoder de Reranking: {config.RERANKER_MODEL}...")
        _reranker_cache = CrossEncoder(config.RERANKER_MODEL)
    return _reranker_cache


def expand_query(query: str) -> str:
    """
    Expande e enriquece a query para aumentar o recall no Estágio 1 (Busca Vetorial).
    Adiciona termos de contexto baseados na intenção detectada na pergunta.
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
    n_retrieval: int = config.TOP_K_RETRIEVAL,
    n_rerank: int = config.TOP_K_RERANK,
    use_reranker: bool = True
) -> list[dict]:
    """
    Executa busca em dois estágios:
    1. Busca Vetorial ampla (Estágio 1 - Bi-Encoder) com query expandida.
    2. Reranking de alta precisão (Estágio 2 - Cross-Encoder) usando a query original do usuário.
    """
    collection = get_collection()
    embed_model = get_embedding_model()
    
    # 1. Expansão de Query
    expanded_query = expand_query(query)
    
    # 2. Prepara texto da query para o modelo Bi-Encoder
    query_text_for_embedding = expanded_query
    if "e5" in config.EMBEDDING_MODEL.lower():
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
        print(f"❌ Erro na busca vetorial no ChromaDB: {e}")
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
        
    # Se reranker não for solicitado ou não houver candidatos suficientes
    if not use_reranker or len(candidates) <= 1:
        return candidates[:n_rerank]
        
    # 4. Reranking com Cross-Encoder (Estágio 2)
    # Sempre usamos a query ORIGINAL (query limpa do usuário) para pontuar a exatidão
    reranker = get_reranker_model()
    pairs = [(query, c["text"]) for c in candidates]
    
    try:
        scores = reranker.predict(pairs)
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(round(score, 4))
            
        # Ordena do maior score (mais relevante) para o menor
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    except Exception as e:
        print(f"⚠️ Aviso ao executar Reranker ({e}). Usando ordenação do ChromaDB.")
        candidates.sort(key=lambda x: x["vector_distance"])
        
    return candidates[:n_rerank]


if __name__ == "__main__":
    # Teste rápido no terminal
    pergunta_teste = "Quais são os procedimentos de leitura e notas do POP?"
    print(f"\n🔍 Testando busca para: '{pergunta_teste}'...")
    trechos = retrieve(pergunta_teste, n_retrieval=8, n_rerank=3)
    for idx, t in enumerate(trechos, 1):
        print(f"\n[{idx}] Página: {t['page_num']} | Score: {t.get('rerank_score', 'N/A')}")
        print(f"Trecho: {t['text'][:200]}...")