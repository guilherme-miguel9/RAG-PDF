import os
import sys

# Define variáveis de ambiente no início do processo para segurança de threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Adiciona a raiz do projeto ao sys.path para importar config e core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from core import document_processor, vector_store


def main():
    print("=" * 65)
    print(" 🛠️  INDEXADOR OFFLINE LANGCHAIN + DOCLING (CLI)")
    print("=" * 65)
    
    force_reindex = "--force" in sys.argv or "-f" in sys.argv
    pdf_path = settings.DEFAULT_PDF_PATH
    
    for arg in sys.argv[1:]:
        if arg.endswith(".pdf") and os.path.exists(arg):
            pdf_path = arg
            break
            
    if not os.path.exists(pdf_path):
        print(f"❌ Erro: Arquivo PDF não encontrado em '{pdf_path}'")
        sys.exit(1)
        
    print(f"📄 Arquivo Alvo: {os.path.basename(pdf_path)}")
    print(f"📁 Banco Vetorial Alvo: {settings.DB_PATH}")
    print(f"📦 Coleção LangChain: {settings.COLLECTION_NAME}")
    print(f"🔄 Forçar Re-indexação: {'Sim' if force_reindex else 'Não'}")
    print("-" * 65)
    
    print("⏳ Iniciando extração profunda de layout e tabelas via DOCLING...")
    count = document_processor.index_pdf(pdf_path, force_reindex=force_reindex, use_docling=True)
    
    print("-" * 65)
    print(f"✅ SUCESSO! Banco vetorial atualizado. Total de fragmentos: {count}")
    print("🤖 O Streamlit agora pode responder instantaneamente sem indexar em tempo de execução.")
    print("=" * 65)


if __name__ == "__main__":
    main()