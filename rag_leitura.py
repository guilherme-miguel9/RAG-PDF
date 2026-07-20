import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import re
import hashlib
from openai import OpenAI
from typing import List, Dict, Tuple

# ==============================
# CONFIGURAÇÕES
# ==============================

PDF_PATH = "pop_leitura.pdf"
DB_PATH = "./memoria_atualizada/"
COLLECTION_NAME = "pdf_leitura"

# Modelo de embeddings (pode ser alterado para um modelo multilíngue se necessário)
#EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDING_MODEL = 'intfloat/multilingual-e5-large'

# Tamanho máximo de cada chunk (em caracteres)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100  # Sobreposição entre chunks consecutivos (útil para manter contexto)

# Cliente para LLM local via LM Studio
client_llm = OpenAI(
    base_url='http://127.0.0.1:1234/v1',
    api_key='lm-studio'
)

# ==============================
# INICIALIZAÇÃO DO CHROMADB
# ==============================

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

# ==============================
# EXTRAÇÃO DE TEXTO COM PÁGINAS
# ==============================

def extract_text_by_page(pdf_path: str) -> List[Dict[str, object]]:
    """
    Extrai o texto de cada página do PDF.
    Retorna uma lista onde cada elemento é um dicionário:
        {'page_num': int, 'text': str}
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():  # ignora páginas completamente vazias
            pages.append({
                'page_num': page_num,
                'text': text
            })
    doc.close()
    return pages

# ==============================
# LIMPEZA SIMPLES DO TEXTO
# ==============================

def clean_text(text: str) -> str:
    """
    Remove espaços excessivos e caracteres de controle,
    mas preserva a estrutura básica e não apaga referências de página.
    """
    # Substitui múltiplos espaços em branco por um único espaço
    text = re.sub(r'\s+', ' ', text)
    # Remove cabeçalhos/rodapés comuns (ex: "Página X | Y") – opcional
    text = re.sub(r'P[aá]gina\s*\d+\s*\|\s*\d+', '', text)
    return text.strip()

# ==============================
# CHUNKING POR PÁGINA (COM SOBREPOSIÇÃO)
# ==============================

def chunk_text(text: str, max_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Divide um texto em chunks de tamanho aproximadamente max_size,
    com sobreposição entre eles para preservar contexto.
    """
    words = text.split()
    chunks = []
    start = 0
    n_words = len(words)
    
    while start < n_words:
        end = min(start + max_size, n_words)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_size - overlap
    return chunks

def process_pages_to_chunks(pages: List[Dict]) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Processa todas as páginas, gera chunks, cria IDs únicos (baseados no texto + página)
    e metadados contendo o número da página.
    
    Retorna:
        all_chunks: lista de textos dos chunks
        all_metadata: lista de dicionários com metadados (incluindo 'page')
        all_ids: lista de IDs únicos para o ChromaDB
    """
    all_chunks = []
    all_metadata = []
    all_ids = []
    
    for page_info in pages:
        page_num = page_info['page_num']
        raw_text = page_info['text']
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue
        
        # Gera chunks para esta página
        page_chunks = chunk_text(cleaned)
        
        for chunk in page_chunks:
            # Cria ID combinando hash do texto e número da página (evita que chunks iguais em páginas diferentes sejam ignorados)
            hash_input = f"{chunk}|page_{page_num}"
            doc_id = hashlib.md5(hash_input.encode()).hexdigest()
            
            all_chunks.append(chunk)
            all_metadata.append({
                "source": PDF_PATH,
                "page": page_num
            })
            all_ids.append(doc_id)
    
    return all_chunks, all_metadata, all_ids

# ==============================
# GERAÇÃO DE EMBEDDINGS E INDEXAÇÃO
# ==============================

def index_pdf(pdf_path: str, force_reindex: bool = False):
    """
    Extrai texto do PDF, gera chunks com página, calcula embeddings e salva no ChromaDB.
    Se force_reindex=True, recria a coleção do zero.
    """
    global collection
    
    if force_reindex:
        client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(COLLECTION_NAME)
        print("Coleção recriada. Indexação forçada.")
    
    print("Extraindo texto por página...")
    pages = extract_text_by_page(pdf_path)
    print(f"Total de páginas com conteúdo: {len(pages)}")
    
    print("Processando páginas em chunks...")
    chunks, metadatas, ids = process_pages_to_chunks(pages)
    print(f"Total de chunks gerados: {len(chunks)}")
    
    if not chunks:
        print("Nenhum conteúdo encontrado no PDF.")
        return
    
    # Verifica quais IDs já existem para evitar reinserção desnecessária
    existing_ids = set()
    try:
        existing = collection.get()
        existing_ids = set(existing["ids"])
    except:
        pass
    
    new_chunks = []
    new_embeddings = []
    new_ids = []
    new_metadatas = []
    
    for i, chunk in enumerate(chunks):
        if ids[i] not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(ids[i])
            new_metadatas.append(metadatas[i])
    
    if new_chunks:
        print(f"Gerando embeddings para {len(new_chunks)} novos chunks...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode(new_chunks)
        
        print(f"Salvando {len(new_chunks)} chunks no banco vetorial...")
        collection.add(
            documents=new_chunks,
            embeddings=[emb.tolist() for emb in embeddings],
            ids=new_ids,
            metadatas=new_metadatas
        )
        print("Indexação concluída.")
    else:
        print("Nenhum novo chunk para adicionar (todos já existentes).")

# ==============================
# CONSULTA COM CONTEXTO ENRIQUECIDO (PÁGINA)
# ==============================

def query_pdf(question: str, n_results: int = 4) -> str:
    """
    Realiza consulta ao banco vetorial, recupera os trechos mais relevantes
    incluindo a página de origem, e gera resposta com LLM.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([question])
    
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    
    # Monta contexto com informação de página
    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        page = meta.get("page", "desconhecida")
        context_parts.append(f"[Página {page}]: {doc}")
    
    context = "\n\n".join(context_parts)
    
    # Prompt com instrução clara para citar páginas quando possível
    system_prompt = (
        "Você é um assistente bom e querido por todos, especializado no conteúdo do PDF fornecido. "
        "Responda apenas com base no contexto abaixo. "
        "Quando o contexto indicar a página (ex: [Página X]), mencione o número da página na sua resposta."
    )
    
    user_prompt = f"Contexto:\n{context}\n\nPergunta: {question}"
    
    response = client_llm.chat.completions.create(
        model='meta-llama-3.1-8b-instruct',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        temperature=0.4  # baixa temperatura para respostas mais precisas
    )
    
    return response.choices[0].message.content

# ==============================
# EXECUÇÃO PRINCIPAL
# ==============================

if __name__ == "__main__":
    # Indexa o PDF (caso já tenha sido indexado antes, não reinsere duplicados)
    index_pdf(PDF_PATH)
    
    # Loop interativo para consultas
    print("\n" + "="*50)
    print("Sistema de Perguntas e Respostas sobre o PDF")
    print("Digite 'sair' para encerrar.")
    print("="*50 + "\n")
    
    while True:
        user_query = input("Sua pergunta: ")
        if user_query.lower() in ['sair', 'exit', 'quit']:
            break
        
        if not user_query.strip():
            continue
        
        print("Consultando...\n")
        resposta = query_pdf(user_query)
        print("Resposta:")
        print(resposta)
        print("\n" + "-"*50 + "\n")