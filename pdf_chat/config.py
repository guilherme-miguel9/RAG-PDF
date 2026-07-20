import os

# =========================================================
# CAMINHOS E DIRETÓRIOS
# =========================================================
# Diretório base do módulo pdf_chat
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Diretório raiz do projeto RAG_LEITURA
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Caminho do banco vetorial ChromaDB (usando o diretório memoria_atualizada na raiz)
DB_PATH = os.path.join(PROJECT_ROOT, "memoria_atualizada")

# Caminho padrão para o PDF original de leitura (se o usuário não fizer upload no Streamlit)
DEFAULT_PDF_PATH = os.path.join(PROJECT_ROOT, "pop_leitura.pdf")

# Nome da coleção no ChromaDB
COLLECTION_NAME = "pdf_rag_producao"

# =========================================================
# MODELOS DE EMBEDDINGS E RERANKING
# =========================================================
# Modelo de Embedding Bi-Encoder para busca vetorial rápida (Estágio 1)
# Opções populares e testadas:
# - 'paraphrase-multilingual-mpnet-base-v2' (Ótimo para PT-BR e semântica geral)
# - 'intfloat/multilingual-e5-base' ou 'intfloat/multilingual-e5-large' (Requer prefixos 'passage: '/'query: ')
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# Modelo de Reranking Cross-Encoder (Estágio 2) para máxima precisão
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# =========================================================
# PARÂMETROS DE CHUNKING E RETRIEVAL
# =========================================================
# Tamanho e sobreposição dos chunks (em caracteres para o divisor semântico)
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

# Quantidade de trechos recuperados no Estágio 1 (Busca Vetorial no ChromaDB)
TOP_K_RETRIEVAL = 12

# Quantidade de trechos selecionados após Reranking no Estágio 2 (enviados ao LLM)
TOP_K_RERANK = 4

# =========================================================
# CONFIGURAÇÕES DO LLM LOCAL (LM STUDIO)
# =========================================================
LLM_BASE_URL = "http://127.0.0.1:1234/v1"
LLM_API_KEY = "lm-studio"
LLM_MODEL_NAME = "meta-llama-3.1-8b-instruct"

# Temperatura (0.1 a 0.3 recomendada para RAG factual sem alucinações)
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048

# =========================================================
# SYSTEM PROMPTS
# =========================================================
SYSTEM_PROMPT = """Você é um assistente técnico especializado no conteúdo do documento/Procedimento Operacional fornecido no contexto.

REGRAS ABSOLUTAS DE COMPORTAMENTO:
1. Responda EXCLUSIVAMENTE com base no texto do documento fornecido abaixo no bloco de CONTEXTO.
2. NUNCA invente, suponha ou utilize conhecimento externo que não esteja explícito no contexto.
3. Se a informação não estiver presente no contexto recuperado, responda EXATAMENTE: "Essa informação não foi encontrada nos trechos recuperados do documento."
4. Citação de Páginas: Sempre mencione a página de onde tirou a informação ao final da frase, item ou parágrafo, no formato (Página X).

ORIENTAÇÕES DE FORMATO:
- LISTAS / ITENS / NOTAS: Quando perguntado sobre listas (ex: quais são os passos, notas, itens, regras), extraia TODOS os itens relevantes presentes no contexto e formate em lista numerada ou com marcadores claros (`- ` ou `1. `).
- PROCEDIMENTOS (HOW-TO): Descreva o passo a passo na ordem lógica em que aparecem no documento.
- EXPLICAÇÕES / DEFINIÇÕES: Sintetize o conceito com clareza, mantendo a precisão técnica dos termos utilizados na leitura."""
