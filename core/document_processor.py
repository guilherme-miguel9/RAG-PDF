import os
import re
import hashlib
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
        print(f"📄 [Processor] Extraindo '{os.path.basename(pdf_path)}' via Docling (Markdown)...")
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
                print(f"✅ [Processor] {len(pages)} páginas convertidas com sucesso via Docling.")
                return pages
        except Exception as e:
            print(f"⚠️ [Processor] Aviso: Falha na extração Docling ({e}). Alternando para PyMuPDF...")

    # Extração de Fallback com PyMuPDF
    print(f"📄 [Processor] Extraindo '{os.path.basename(pdf_path)}' com PyMuPDF (fitz)...")
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
    print(f"✅ [Processor] {len(pages)} páginas extraídas via PyMuPDF.")
    return pages


# =========================================================
# 2. CHUNKING SEMÂNTICO INTELIGENTE
# =========================================================

def semantic_chunking(pages: list[dict], chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP) -> tuple[list, list, list]:
    """
    Divide o texto em fragmentos respeitando limites de parágrafos e frases sem cortar palavras.
    Gera metadados contendo o número exato da página de origem.
    """
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
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
                    _add_chunk(chunk_text, page_num, all_chunks, all_metadatas, all_ids)
                    
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
                    _split_long_paragraph(para, page_num, chunk_size, overlap, all_chunks, all_metadatas, all_ids)
                    buffer = []
                    buffer_len = 0
        
        if buffer:
            chunk_text = "\n\n".join(buffer).strip()
            _add_chunk(chunk_text, page_num, all_chunks, all_metadatas, all_ids)
            
    return all_chunks, all_metadatas, all_ids


def _split_long_paragraph(text: str, page_num: int, max_size: int, overlap: int, chunks: list, metadatas: list, ids: list):
    """Auxiliar para quebrar parágrafos muito extensos sem cortar palavras ao meio."""
    words = text.split()
    current = []
    current_size = 0
    
    for w in words:
        if current_size + len(w) + 1 > max_size and current:
            chunk_str = " ".join(current)
            _add_chunk(chunk_str, page_num, chunks, metadatas, ids)
            
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
# 3. ROTINA PRINCIPAL DE INDEXAÇÃO
# =========================================================

def index_pdf(pdf_path: str, force_reindex: bool = False) -> int:
    """
    Orquestra a extração do PDF, fragmentação semântica e gravação vetorial no ChromaDB.
    """
    if force_reindex:
        vector_store.delete_collection()

    collection = vector_store.get_collection()
    
    pages = extract_pages(pdf_path, use_docling=True)
    if not pages:
        print("❌ [Processor] Nenhum texto foi identificado no PDF.")
        return 0
        
    chunks, metadatas, ids = semantic_chunking(pages, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    print(f"✂️ [Processor] Gerados {len(chunks)} blocos semânticos de {len(pages)} páginas.")
    
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
        print(f"🧠 [Processor] Computando embeddings para {len(new_chunks)} novos blocos...")
        model = vector_store.get_embedding_model()
        
        texts_to_encode = new_chunks
        if "e5" in settings.EMBEDDING_MODEL.lower():
            texts_to_encode = [f"passage: {t}" for t in new_chunks]
            
        embeddings = model.encode(
            texts_to_encode,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        ).tolist()
        
        print(f"💾 [Processor] Armazenando no ChromaDB...")
        batch_size = 500
        for b in range(0, len(new_chunks), batch_size):
            collection.add(
                ids=new_ids[b:b+batch_size],
                documents=new_chunks[b:b+batch_size],
                metadatas=new_metadatas[b:b+batch_size],
                embeddings=embeddings[b:b+batch_size]
            )
        print("✅ [Processor] Indexação concluída com sucesso.")
    else:
        print("⚡ [Processor] Documento já indexado integralmente no ChromaDB.")
        
    return collection.count()