from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import hashlib
import sys
import re

# ========== CONFIGURAÇÕES ==========
PDF_PATH = "pop_leitura.pdf"
DB_PATH = "./memoria_atualizada/"
COLLECTION_NAME = "pop_leitura_chunks_v2"

EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
RERANKER_MODEL  = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

TOP_K_RETRIEVAL = 10
TOP_K_RERANK    = 3   # mais contexto para interpretar

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 120

# ========== INICIALIZAÇÃO DO LLM ==========
llm_leitura = OpenAI(base_url='http://127.0.0.1:1234/v1', api_key='llm')


# ========== EXTRAÇÃO COM DOCLING ==========
def extract_pages_docling(pdf_path: str) -> list[dict]:
    print("📄 Extraindo PDF com Docling...")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend
            )
        }
    )

    result = converter.convert(pdf_path)
    doc    = result.document
    pages  = []

    for element, _level in doc.iterate_items():
        text = element.text.strip() if hasattr(element, "text") else ""
        if not text:
            continue

        page_num = None
        if hasattr(element, "prov") and element.prov:
            page_num = element.prov[0].page_no

        if page_num is None:
            page_num = 1

        pages.append({"page_num": page_num, "text": text})

    print(f"✅ {len(pages)} blocos extraídos via Docling.")
    return pages


# ========== CHUNKING SEMÂNTICO ==========
def semantic_chunking(pages: list[dict], chunk_size: int, overlap: int) -> tuple[list, list, list]:
    all_chunks    = []
    all_metadatas = []
    all_ids       = []

    for page_data in pages:
        page_num  = page_data["page_num"]
        page_text = page_data["text"]

        paragraphs = re.split(r"\n{2,}", page_text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        buffer = ""
        for para in paragraphs:
            if len(buffer) + len(para) + 1 <= chunk_size:
                buffer = (buffer + "\n\n" + para).strip()
            else:
                if buffer:
                    _add_chunk(buffer, page_num, all_chunks, all_metadatas, all_ids)
                    buffer = buffer[-overlap:] + "\n\n" + para
                else:
                    buffer = para

        if buffer:
            _add_chunk(buffer, page_num, all_chunks, all_metadatas, all_ids)

    print(f"✂️  {len(all_chunks)} chunks gerados (chunking semântico).")
    return all_chunks, all_metadatas, all_ids


def _add_chunk(text, page_num, chunks, metadatas, ids):
    chunk_id = hashlib.md5(f"{page_num}_{text[:60]}".encode()).hexdigest()
    chunks.append(text)
    metadatas.append({"page_num": page_num})
    ids.append(chunk_id)


# ========== INDEXAÇÃO ==========
def index_pdf(force_reindex=False):
    client   = chromadb.PersistentClient(path=DB_PATH)
    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing and not force_reindex:
        print(f"✅ Coleção '{COLLECTION_NAME}' encontrada. Carregando...")
        collection = client.get_collection(COLLECTION_NAME)
        return collection, None

    if force_reindex:
        print("🔄 Forçando reindexação...")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    pages = extract_pages_docling(PDF_PATH)
    all_chunks, all_metadatas, all_ids = semantic_chunking(pages, CHUNK_SIZE, CHUNK_OVERLAP)

    print(f"🧠 Gerando embeddings com '{EMBEDDING_MODEL}'...")
    model      = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        all_chunks,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True
    )

    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=all_ids,
        documents=all_chunks,
        metadatas=all_metadatas,
        embeddings=embeddings.tolist()
    )
    print(f"✅ {collection.count()} chunks indexados.\n")
    return collection, model


# ========== EXPANSÃO DE QUERY ==========
def expand_query(query: str) -> str:
    """
    Reformula a query para recuperar chunks mais relevantes.
    Detecta a intenção do usuário e adiciona termos relacionados.
    """
    expansions = {
        r"list[ae]|liste|quais (são os|os|as)|enumere": "lista itens enumeração",
        r"o que [eé]|explique|defina|definição|significa":  "definição conceito descrição explicação",
        r"como (fazer|realizar|executar|funciona|é feito)": "procedimento passo etapa modo execução",
        r"procedimento|protocolo|processo":                 "procedimento protocolo etapas instruções passos",
        r"nota[s]?|observa[cç][aã]o|aviso":                "nota observação aviso atenção importante",
        r"objetivo[s]?|finalidade|para que":                "objetivo finalidade propósito meta",
        r"responsável|quem (deve|faz|realiza)":             "responsável executor função cargo",
    }
    expanded = query
    for pattern, extra in expansions.items():
        if re.search(pattern, query, re.IGNORECASE):
            expanded = f"{query} {extra}"
            break
    return expanded


# ========== CONSULTA COM RERANKING ==========
def query_pdf(collection, embed_model, reranker, user_query: str) -> str:

    # 1. Expande a query para recuperar chunks mais relevantes
    expanded_query = expand_query(user_query)

    # 2. Embedding da query expandida
    query_vec = embed_model.encode(
        [expanded_query], normalize_embeddings=True
    )[0]

    # 3. Recuperação ampla
    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=TOP_K_RETRIEVAL,
        include=["documents", "metadatas", "distances"]
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]

    # 4. Reranking com a query ORIGINAL (fiel à intenção real)
    pairs  = [(user_query, doc) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, docs, metadatas),
        key=lambda x: x[0],
        reverse=True
    )[:TOP_K_RERANK]

    # 5. Monta contexto
    context = "\n\n".join(
        f"[Página {m['page_num']}]\n{d}"
        for _, d, m in ranked
    )

    # 6. Prompt focado em INTERPRETAR e EXPLICAR
    system_prompt = """Você é um assistente que responde EXCLUSIVAMENTE com base no texto do documento abaixo.

REGRAS ABSOLUTAS — siga sem exceção:
- NUNCA invente, suponha ou use conhecimento externo ao documento.
- Se uma informação não aparecer textualmente no contexto abaixo, ela NÃO existe para você.
- PROIBIDO completar, deduzir ou extrapolar dados não presentes no contexto.
- Se não encontrar a informação, responda EXATAMENTE: "Essa informação não está presente no trecho recuperado do documento."

Comportamento por tipo de solicitação:

- LISTAS (ex: "liste as notas", "quais os itens"):
  → Extraia TODOS os itens encontrados no contexto e organize em lista numerada ou com marcadores, clara e completa.

- EXPLICAÇÕES (ex: "o que é X", "explique X", "o que significa X"):
  → Explique com clareza o que o documento diz sobre o assunto, sintetizando com suas próprias palavras. Seja detalhado.

- PROCEDIMENTOS (ex: "como fazer X", "como é feito X"):
  → Descreva passo a passo conforme o documento, em ordem lógica.

- GERAL:
  → Interprete e sintetize o conteúdo relevante. Nunca copie o texto cru — sempre organize e explique.
  → Mencione a página de forma discreta ao final de cada item ou resposta: (Página X).
  → Use todo o contexto disponível para dar a resposta mais completa possível."""

    response = llm_leitura.chat.completions.create(
        model="meta-llama-3.1-8b-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Contexto do documento:\n{context}\n\nSolicitação: {user_query}"}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


# ========== MAIN ==========
if __name__ == "__main__":
    force = "--force-reindex" in sys.argv

    collection, embed_model = index_pdf(force_reindex=force)

    if embed_model is None:
        print("Carregando modelo de embedding...")
        embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Carregando reranker...")
    reranker = CrossEncoder(RERANKER_MODEL)

    print("\n" + "=" * 50)
    print("Sistema RAG com PDF — versão melhorada")
    print("Digite 'sair' para encerrar.")
    print("=" * 50 + "\n")

    while True:
        user_query = input("Sua pergunta: ").strip()
        if user_query.lower() in ["sair", "exit", "quit"]:
            break
        if not user_query:
            continue

        print("Consultando...\n")
        resposta = query_pdf(collection, embed_model, reranker, user_query)
        print(f"Resposta:\n{resposta}")
        print("\n" + "-" * 50 + "\n")