import os

# Define variáveis de ambiente para evitar conflitos de threads (OpenMP/MKL/ONNX) em workers no Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import hashlib
from langchain_core.documents import Document
from config import settings
from core import vector_store

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False

import fitz  # PyMuPDF (fallback rápido ou alternativo)


# =========================================================
# 1. EXTRAÇÃO DE TEXTO DO PDF (POR PÁGINA EM MARKDOWN)
# =========================================================

def extract_pages(pdf_path: str, use_docling: bool = True) -> list[dict]:
    """
    Extrai o texto por página de um PDF.
    Utiliza o Docling para conversão em Markdown estruturado (tabelas e listas preservadas).
    Aciona fallback automático ao PyMuPDF em caso de indisponibilidade ou erro.
    """
    pages = []
    
    if use_docling and HAS_DOCLING:
        print(f"[LangChain Processor] Extraindo '{os.path.basename(pdf_path)}' via Docling (Markdown)...")
        try:
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
            doc = result.document
            
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
                print(f"[LangChain Processor] {len(pages)} páginas convertidas via Docling com sucesso.")
                return pages
        except Exception as e:
            print(f"[LangChain Processor] Aviso na extração Docling ({e}). Alternando para PyMuPDF...")

    # Extração de Fallback com PyMuPDF
    print(f"[LangChain Processor] Extraindo '{os.path.basename(pdf_path)}' com PyMuPDF (fitz)...")
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            pages.append({
                "page_num": i + 1,
                "text": text
            })
    doc.close()
    print(f"[LangChain Processor] {len(pages)} páginas extraídas via PyMuPDF.")
    return pages


# =========================================================
# 2. CHUNKING SEMÂNTICO EM OBJETOS LANGCHAIN DOCUMENT
# =========================================================

def semantic_chunking_to_documents(pages: list[dict], chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP) -> list[Document]:
    """
    Divide o texto em fragmentos respeitando limites de parágrafos e frases sem cortar palavras.
    Converte cada fragmento diretamente em objetos Document oficiais do LangChain.
    """
    chunks_text = []
    chunks_meta = []
    chunks_id = []
    
    for page_data in pages:
        page_num = page_data["page_num"]
        page_text = page_data["text"]
        
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
                    _add_raw_chunk(chunk_text, page_num, chunks_text, chunks_meta, chunks_id)
                    
                    if len(buffer[-1]) <= overlap:
                        buffer = [buffer[-1], para]
                        buffer_len = len(buffer[0]) + para_len + 2
                    else:
                        last_text = buffer[-1][-overlap:]
                        space_idx = last_text.find(" ")
                        if space_idx != -1:
                            last_text = last_text[space_idx:].strip()
                        buffer = [last_text, para]
                        buffer_len = len(last_text) + para_len + 2
                else:
                    _split_long_paragraph(para, page_num, chunk_size, overlap, chunks_text, chunks_meta, chunks_id)
                    buffer = []
                    buffer_len = 0
        
        if buffer:
            chunk_text = "\n\n".join(buffer).strip()
            _add_raw_chunk(chunk_text, page_num, chunks_text, chunks_meta, chunks_id)
            
    langchain_docs = []
    for c_text, c_meta, c_id in zip(chunks_text, chunks_meta, chunks_id):
        langchain_docs.append(Document(
            page_content=c_text,
            metadata={
                "page_num": c_meta["page_num"],
                "source": c_meta["source"],
                "doc_id": c_id
            }
        ))
    return langchain_docs


def _split_long_paragraph(text: str, page_num: int, max_size: int, overlap: int, chunks: list, metadatas: list, ids: list):
    """Auxiliar para quebrar parágrafos muito extensos sem cortar palavras ao meio."""
    words = text.split()
    current = []
    current_size = 0
    
    for w in words:
        if current_size + len(w) + 1 > max_size and current:
            chunk_str = " ".join(current)
            _add_raw_chunk(chunk_str, page_num, chunks, metadatas, ids)
            
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
        _add_raw_chunk(" ".join(current), page_num, chunks, metadatas, ids)


def _add_raw_chunk(text: str, page_num: int, chunks: list, metadatas: list, ids: list):
    """Grava o chunk e gera seu hash MD5 para unicidade."""
    if not text.strip():
        return
    chunk_id = hashlib.md5(f"p{page_num}_{text[:100]}".encode("utf-8")).hexdigest()
    chunks.append(text.strip())
    metadatas.append({
        "page_num": page_num,
        "source": settings.COLLECTION_NAME
    })
    ids.append(chunk_id)


# =========================================================
# 3. ROTINA PRINCIPAL DE INDEXAÇÃO LANGCHAIN
# =========================================================

def index_pdf(pdf_path: str, force_reindex: bool = False, use_docling: bool = False) -> int:
    """
    Orquestra a extração do PDF, fragmentação em objetos Document e gravação no Chroma LangChain.
    Se use_docling=True (recomendado para CLI offline), extrai tabelas e layout via Docling.
    Se use_docling=False (recomendado para Streamlit/UI rápida), utiliza PyMuPDF (fitz).
    """
    if force_reindex:
        vector_store.delete_collection()

    db = vector_store.get_vector_db()
    
    pages = extract_pages(pdf_path, use_docling=use_docling)
    if not pages:
        print("[LangChain Processor] Nenhum texto foi identificado no PDF.")
        return 0
        
    langchain_docs = semantic_chunking_to_documents(pages, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    print(f"[LangChain Processor] Gerados {len(langchain_docs)} documentos LangChain de {len(pages)} páginas.")
    
    existing_ids = set()
    try:
        existing = db._collection.get(include=[])
        existing_ids = set(existing["ids"])
    except Exception:
        pass
        
    new_docs = []
    new_ids = []
    
    for doc in langchain_docs:
        doc_id = doc.metadata.get("doc_id")
        if doc_id and doc_id not in existing_ids:
            new_docs.append(doc)
            new_ids.append(doc_id)
            existing_ids.add(doc_id)
            
    if new_docs:
        print(f"[LangChain Processor] Armazenando {len(new_docs)} novos documentos na base Chroma LangChain...")
        batch_size = 500
        for b in range(0, len(new_docs), batch_size):
            batch_docs = new_docs[b:b+batch_size]
            batch_ids = new_ids[b:b+batch_size]
            db.add_documents(documents=batch_docs, ids=batch_ids)
        print("[LangChain Processor] Indexação LangChain concluída com sucesso.")
    else:
        print("[LangChain Processor] Documentos já presentes na base LangChain.")
        
    return db._collection.count()