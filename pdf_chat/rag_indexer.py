import os
import re
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False

import fitz  # PyMuPDF (sempre disponível como fallback ou principal)
import config

# Inicialização do cliente ChromaDB
client = chromadb.PersistentClient(path=config.DB_PATH)

# Cache em memória do modelo de embedding para evitar recarregamento
_model_cache = None


def get_embedding_model():
    """Carrega o modelo de embeddings (com cache em memória)."""
    global _model_cache
    if _model_cache is None:
        print(f"🧠 Carregando modelo de embedding: {config.EMBEDDING_MODEL}...")
        _model_cache = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model_cache


def get_collection():
    """Obtém ou cria a coleção no ChromaDB com métrica de distância cosseno."""
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def get_collection_stats() -> dict:
    """Retorna estatísticas da coleção para exibição na interface."""
    try:
        coll = client.get_collection(config.COLLECTION_NAME)
        count = coll.count()
        return {
            "exists": True,
            "count": count,
            "collection_name": config.COLLECTION_NAME,
            "db_path": config.DB_PATH
        }
    except Exception:
        return {
            "exists": False,
            "count": 0,
            "collection_name": config.COLLECTION_NAME,
            "db_path": config.DB_PATH
        }


# =========================================================
# 1. EXTRAÇÃO DE TEXTO DO PDF (POR PÁGINA COM MARKDOWN/OCR OPT)
# =========================================================

def extract_pages(pdf_path: str, use_docling: bool = True) -> list[dict]:
    """
    Extrai o texto por página de um PDF.
    Tenta usar Docling para conversão estruturada em Markdown (preserva tabelas e listas).
    Usa PyMuPDF como fallback rápido ou se Docling não estiver disponível.
    """
    pages = []
    
    if use_docling and HAS_DOCLING:
        print(f"📄 Extraindo '{os.path.basename(pdf_path)}' estruturado com Docling (Markdown)...")
        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False  # Para PDFs nativos, OCR desativado é mais rápido
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        backend=PyPdfiumDocumentBackend
                    )
                }
            )
            
            # Converte e itera pelos itens para mapear por página
            result = converter.convert(pdf_path)
            doc = result.document
            
            # Vamos mapear texto por página
            page_dict = {}
            for element, _level in doc.iterate_items():
                text = element.text.strip() if hasattr(element, "text") else ""
                if not text:
                    continue
                
                page_num = 1
                if hasattr(element, "prov") and element.prov:
                    page_num = element.prov[0].page_no
                
                if page_num not in page_dict:
                    page_dict[page_num] = []
                page_dict[page_num].append(text)
            
            for p_num in sorted(page_dict.keys()):
                pages.append({
                    "page_num": p_num,
                    "text": "\n\n".join(page_dict[p_num])
                })
            
            if pages:
                print(f"✅ {len(pages)} páginas extraídas com sucesso via Docling.")
                return pages
        except Exception as e:
            print(f"⚠️ Aviso: Erro no Docling ({e}). Alternando para fallback PyMuPDF...")

    # Fallback / Extração com PyMuPDF
    print(f"📄 Extraindo '{os.path.basename(pdf_path)}' com PyMuPDF (fitz)...")
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            # Limpeza leve preservando quebras de parágrafo
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            pages.append({
                "page_num": i + 1,
                "text": text
            })
    doc.close()
    print(f"✅ {len(pages)} páginas extraídas via PyMuPDF.")
    return pages


# =========================================================
# 2. CHUNKING SEMÂNTICO (POR PARÁGRAFOS COM OVERLAP SEGURO)
# =========================================================

def semantic_chunking(pages: list[dict], chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> tuple[list, list, list]:
    """
    Divide o texto em chunks respeitando limites semânticos (parágrafos/frases),
    sem cortar palavras no meio, e mantendo metadados detalhados de página.
    """
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    for page_data in pages:
        page_num = page_data["page_num"]
        page_text = page_data["text"]
        
        # Divide por parágrafos duplos ou quebras de linha com listas
        paragraphs = re.split(r"\n{2,}", page_text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        buffer = []
        buffer_len = 0
        
        for para in paragraphs:
            para_len = len(para)
            
            if buffer_len + para_len + 2 <= chunk_size:
                buffer.append(para)
                buffer_len += para_len + 2
            else:
                if buffer:
                    chunk_text = "\n\n".join(buffer).strip()
                    _add_chunk(chunk_text, page_num, all_chunks, all_metadatas, all_ids)
                    
                    # Overlap inteligente: pega o último parágrafo inteiro ou últimas frases
                    if len(buffer[-1]) <= overlap:
                        buffer = [buffer[-1], para]
                        buffer_len = len(buffer[0]) + para_len + 2
                    else:
                        # Se o último parágrafo for maior que overlap, pega as últimas palavras/overlap dele
                        last_text = buffer[-1][-overlap:]
                        # Acha o primeiro espaço para não cortar palavra
                        space_idx = last_text.find(" ")
                        if space_idx != -1:
                            last_text = last_text[space_idx:].strip()
                        buffer = [last_text, para]
                        buffer_len = len(last_text) + para_len + 2
                else:
                    # Parágrafo único gigante (> chunk_size): divide por frases ou blocos de palavras
                    _split_long_paragraph(para, page_num, chunk_size, overlap, all_chunks, all_metadatas, all_ids)
                    buffer = []
                    buffer_len = 0
        
        if buffer:
            chunk_text = "\n\n".join(buffer).strip()
            _add_chunk(chunk_text, page_num, all_chunks, all_metadatas, all_ids)
            
    return all_chunks, all_metadatas, all_ids


def _split_long_paragraph(text: str, page_num: int, max_size: int, overlap: int, chunks: list, metadatas: list, ids: list):
    """Auxiliar para fatiar parágrafos muito longos sem cortar palavras no meio."""
    words = text.split()
    current = []
    current_size = 0
    
    for w in words:
        if current_size + len(w) + 1 > max_size and current:
            chunk_str = " ".join(current)
            _add_chunk(chunk_str, page_num, chunks, metadatas, ids)
            
            # Mantém as últimas N palavras para o overlap
            overlap_words = []
            overlap_size = 0
            for ow in reversed(current):
                if overlap_size + len(ow) + 1 <= overlap:
                    overlap_words.insert(0, ow)
                    overlap_size += len(ow) + 1
                else:
                    break
            current = overlap_words[:]
            current_size = overlap_size
            
        current.append(w)
        current_size += len(w) + 1
        
    if current:
        _add_chunk(" ".join(current), page_num, chunks, metadatas, ids)


def _add_chunk(text: str, page_num: int, chunks: list, metadatas: list, ids: list):
    """Adiciona chunk à lista, gerando ID único baseado em hash MD5 do texto e página."""
    if not text.strip():
        return
    # Hash md5 para unicidade e deduplicação
    chunk_id = hashlib.md5(f"p{page_num}_{text[:100]}".encode("utf-8")).hexdigest()
    chunks.append(text.strip())
    metadatas.append({
        "page_num": page_num,
        "source": config.COLLECTION_NAME
    })
    ids.append(chunk_id)


# =========================================================
# 3. INDEXAÇÃO PRINCIPAL (CHROMADB)
# =========================================================

def index_pdf(pdf_path: str, force_reindex: bool = False) -> int:
    """
    Processa o PDF, divide em chunks semânticos, calcula embeddings e salva no ChromaDB.
    Se force_reindex for True, limpa e recria a coleção do zero.
    Retorna o número total de trechos indexados na coleção.
    """
    if force_reindex:
        print(f"🔄 Forçando recriação da coleção '{config.COLLECTION_NAME}'...")
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass

    collection = get_collection()
    
    # Extrai páginas
    pages = extract_pages(pdf_path, use_docling=True)
    if not pages:
        print("❌ Nenhum texto foi extraído do PDF.")
        return 0
        
    # Gera chunks
    chunks, metadatas, ids = semantic_chunking(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"✂️ Gerados {len(chunks)} chunks semânticos de {len(pages)} páginas.")
    
    # Verifica quais chunks já estão no banco para indexação incremental
    existing_ids = set()
    try:
        existing = collection.get(include=[])
        existing_ids = set(existing["ids"])
    except Exception:
        pass
        
    new_chunks = []
    new_metadatas = []
    new_ids = []
    
    for c, m, i in zip(chunks, metadatas, ids):
        if i not in existing_ids:
            new_chunks.append(c)
            new_metadatas.append(m)
            new_ids.append(i)
            
    if new_chunks:
        print(f"🧠 Gerando embeddings para {len(new_chunks)} novos chunks com o modelo '{config.EMBEDDING_MODEL}'...")
        model = get_embedding_model()
        
        # Se for modelo E5, adiciona prefixo 'passage: ' aos documentos na indexação
        texts_to_encode = new_chunks
        if "e5" in config.EMBEDDING_MODEL.lower():
            texts_to_encode = [f"passage: {t}" for t in new_chunks]
            
        embeddings = model.encode(
            texts_to_encode,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        ).tolist()
        
        print(f"💾 Salvando {len(new_chunks)} novos chunks na coleção ChromaDB...")
        # Lote de inserções se for muito grande
        batch_size = 500
        for b in range(0, len(new_chunks), batch_size):
            collection.add(
                ids=new_ids[b:b+batch_size],
                documents=new_chunks[b:b+batch_size],
                metadatas=new_metadatas[b:b+batch_size],
                embeddings=embeddings[b:b+batch_size]
            )
        print("✅ Indexação incremental concluída com sucesso.")
    else:
        print("⚡ Todos os chunks do PDF já estavam indexados no ChromaDB.")
        
    return collection.count()


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv or "-f" in sys.argv
    pdf = config.DEFAULT_PDF_PATH
    if os.path.exists(pdf):
        n = index_pdf(pdf, force_reindex=force)
        print(f"\n✨ Total de chunks no banco: {n}")
    else:
        print(f"❌ Arquivo PDF não encontrado em: {pdf}")