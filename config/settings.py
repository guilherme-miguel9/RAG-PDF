import os

# =========================================================
# CAMINHOS E DIRETÓRIOS DO PROJETO (ARQUITETURA LIMPA)
# =========================================================
# Diretório base (config/)
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Diretório raiz do projeto RAG-PDF
PROJECT_ROOT = os.path.abspath(os.path.join(CONFIG_DIR, ".."))

# Diretórios estruturados de dados (protegidos no .gitignore)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
DB_PATH = os.path.join(DATA_DIR, "chroma_db")

# Caminho padrão para o PDF original do POP (em data/raw/pop_leitura.pdf)
DEFAULT_PDF_PATH = os.path.join(RAW_DATA_DIR, "pop_leitura.pdf")

# Nome da coleção no banco ChromaDB
COLLECTION_NAME = "pdf_rag_producao"

# =========================================================
# MODELOS DE EMBEDDINGS E RERANKING
# =========================================================
# Estágio 1 - Busca Vetorial Rápida (Bi-Encoder)
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# Estágio 2 - Reranking de Alta Precisão (Cross-Encoder)
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# =========================================================
# PARÂMETROS DE DIVISÃO E RECUPERAÇÃO DE TEXTO
# =========================================================
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

TOP_K_RETRIEVAL = 12
TOP_K_RERANK = 4

# =========================================================
# CONFIGURAÇÕES DO SERVIDOR LLM LOCAL (LM STUDIO)
# =========================================================
LLM_BASE_URL = "http://127.0.0.1:1234/v1"
LLM_API_KEY = "lm-studio"
LLM_MODEL_NAME = "meta-llama-3.1-8b-instruct"

LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048

# =========================================================
# PROMPTS E INSTRUÇÕES TÉCNICAS
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
