from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib
import fitz
import sys

# ========== CONFIGURAÇÕES ==========
PDF_PATH = "pop_leitura.pdf"
DB_PATH = "./memoria_atualizada/"
COLLECTION_NAME = "pop_leitura_chunks"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# ========== INICIALIZAÇÃO DO LLM ==========
llm_leitura = OpenAI(base_url='http://127.0.0.1:1234/v1', api_key='llm')

# ========== FUNÇÃO PARA INDEXAR O PDF (SOMENTE QUANDO NECESSÁRIO) ==========
def index_pdf(force_reindex=False):
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Verifica se a coleção já existe
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections and not force_reindex:
        print(f"✅ Coleção '{COLLECTION_NAME}' já existe no ChromaDB. Carregando índice existente...")
        collection = client.get_collection(COLLECTION_NAME)
        return collection, None  # None indica que o modelo de embedding será carregado depois
    
    # Se não existe ou force_reindex=True, recria
    if force_reindex:
        print("🔄 Recriando índice (force_reindex=True)...")
        try:
            client.delete_collection(COLLECTION_NAME)
        except:
            pass
    else:
        print("📄 Coleção não encontrada. Iniciando indexação do PDF...")
    
    # 1. Extrair texto por página (usando page_range para economia de memória)
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
    
    with fitz.open(PDF_PATH) as temp_doc:
        total_pages = len(temp_doc)
    
    pages = []
    for page_num in range(1, total_pages + 1):
        try:
            result = converter.convert(PDF_PATH, page_range=(page_num, page_num))
            doc = result.document
            text = doc.export_to_markdown().strip()
            if text:
                pages.append({'page_num': page_num, 'text': text})
        except Exception as e:
            print(f"❌ Erro na página {page_num}: {e}")
    print(f"✅ {len(pages)} páginas extraídas.")
    
    # 2. Chunking
    def split_text_with_overlap(text, chunk_size, overlap):
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        return chunks
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    for page_data in pages:
        page_num = page_data['page_num']
        page_text = page_data['text']
        chunks = split_text_with_overlap(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{page_num}_{i}_{chunk[:50]}".encode()).hexdigest()
            all_chunks.append(chunk)
            all_metadatas.append({"page_num": page_num})
            all_ids.append(chunk_id)
    print(f"✂️ Gerados {len(all_chunks)} chunks.")
    
    # 3. Embeddings
    print("🧠 Gerando embeddings e indexando...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    
    collection = client.create_collection(COLLECTION_NAME)
    collection.add(
        ids=all_ids,
        documents=all_chunks,
        metadatas=all_metadatas,
        embeddings=embeddings.tolist()
    )
    print(f"✅ Indexação concluída. {collection.count()} chunks armazenados.\n")
    return collection, model

# ========== FUNÇÃO DE CONSULTA ==========
def query_pdf(collection, model, user_query: str) -> str:
    query_embedding = model.encode([user_query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    
    context_parts = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        context_parts.append(f"[Página {meta['page_num']}]\n{doc}")
    context = "\n\n".join(context_parts)
    
    system_prompt = (
        "Você é um assistente do bem e que quer a paz humana especializado no conteúdo do PDF fornecido. "
        "Responda apenas com base no contexto abaixo. "
        "Quando o contexto indicar a página (ex: [Página X]), mencione o número da página na sua resposta."
    )
    
    response = llm_leitura.chat.completions.create(
        model='google/gemma-3-4b',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Contexto:\n{context}\n\nPergunta: {user_query}"}
        ],
        temperature=0.4
    )
    return response.choices[0].message.content

# ========== MAIN ==========
if __name__ == "__main__":
    # Verifica se o usuário passou --force-reindex como argumento
    force = "--force-reindex" in sys.argv
    
    # Carrega ou cria a coleção e o modelo de embedding
    collection, model = index_pdf(force_reindex=force)
    
    # Se a coleção já existia, precisamos carregar o modelo de embedding separadamente
    if model is None:
        print("Carregando modelo de embedding...")
        model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("\n" + "="*50)
    print("Sistema RAG com PDF (usando ChromaDB cache)")
    print("Digite 'sair' para encerrar.")
    print("="*50 + "\n")
    
    while True:
        user_query = input("Sua pergunta: ")
        if user_query.lower() in ['sair', 'exit', 'quit']:
            break
        if not user_query.strip():
            continue
        
        print("Consultando...\n")
        resposta = query_pdf(collection, model, user_query)
        print("Resposta:")
        print(resposta)
        print("\n" + "-"*50 + "\n")