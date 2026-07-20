import os

# Define variáveis de ambiente no início do processo para evitar falhas nativas de OpenMP/ONNX no Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import tempfile
import streamlit as st

from config import settings
from core import document_processor, vector_store, retriever, generator

# =========================================================
# CONFIGURAÇÃO DA PÁGINA E DESIGN SYSTEM MINIMALISTA
# =========================================================
st.set_page_config(
    page_title="RAG — Assistente de Procedimentos Operacionais",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Customização CSS Avançada (Design Minimalista, Premium e Responsivo)
st.markdown("""
<style>
    /* Tipografia de Sistema de Alta Precisão */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
        letter-spacing: -0.01em;
    }

    /* Redução de margens superiores e padding responsivo */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        max-width: 1080px;
    }

    /* Títulos e Subtítulos Minimalistas */
    h1 {
        font-weight: 700;
        font-size: 2.2rem !important;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem !important;
    }
    h2, h3 {
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    p, span, div {
        line-height: 1.55;
    }

    /* Cards e Superfícies (Glassmorphism sutil e bordas refinadas) */
    .stApp {
        background-color: transparent;
    }
    
    /* Customização de Botões (Estilo Pill e Feedback Tátil) */
    .stButton > button {
        border-radius: 9999px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.4rem !important;
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
        background: rgba(128, 128, 128, 0.05) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03) !important;
    }
    .stButton > button:hover {
        background: rgba(128, 128, 128, 0.12) !important;
        border-color: rgba(128, 128, 128, 0.4) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
    }

    /* Sidebar minimalista e estruturada */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.15);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem !important;
    }

    /* Blocos de Trecho Recuperado (Cards de Contexto) */
    .context-card {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 16px;
        margin-top: 10px;
        margin-bottom: 12px;
        font-size: 0.92rem;
    }
    .context-header {
        font-weight: 600;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.75;
        margin-bottom: 8px;
    }

    /* Entrada de chat flutuante minimalista */
    .stChatInput > div {
        border-radius: 9999px !important;
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
        background: rgba(128, 128, 128, 0.04) !important;
    }
    .stChatInput > div:focus-within {
        border-color: rgba(128, 128, 128, 0.6) !important;
        box-shadow: 0 0 0 2px rgba(128, 128, 128, 0.15) !important;
    }

    /* Expanders limpos sem bordas pesadas */
    .streamlit-expanderHeader {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# GESTÃO DE ESTADO DA SESSÃO
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "stats" not in st.session_state:
    st.session_state.stats = vector_store.get_collection_stats()


def update_stats():
    """Atualiza as estatísticas operacionais do banco vetorial no estado da sessão."""
    st.session_state.stats = vector_store.get_collection_stats()


# =========================================================
# BARRA LATERAL (PAINEL DE CONTROLE)
# =========================================================
with st.sidebar:
    st.markdown("### Painel de Sistema")
    st.caption("Arquitetura RAG Bi-Encoder & Cross-Encoder")
    st.divider()
    
    # --- STATUS DA BASE VETORIAL ---
    st.markdown("#### Base de Conhecimento")
    stats = st.session_state.stats
    if stats["count"] > 0:
        st.markdown(f"**Status:** Ativo (`{stats['collection_name']}`)")
        st.metric(label="Fragmentos Indexados", value=stats["count"])
    else:
        st.markdown("**Status:** Base não indexada")
        st.metric(label="Fragmentos Indexados", value=0)
        
    st.divider()
    
    # --- INGESTÃO DE DOCUMENTOS ---
    st.markdown("#### Ingestão de Documentos")
    
    if os.path.exists(settings.DEFAULT_PDF_PATH):
        if st.button("Indexar Documento Padrão", use_container_width=True):
            with st.status("Processando documento...", expanded=True) as status:
                st.write("Convertendo páginas estruturadas via Docling...")
                st.write("Aplicando segmentação semântica...")
                n = document_processor.index_pdf(settings.DEFAULT_PDF_PATH, force_reindex=False)
                status.update(label="Indexação concluída com sucesso", state="complete", expanded=False)
            update_stats()
            st.rerun()
    else:
        st.caption(f"Documento padrão não localizado em: data/raw/{os.path.basename(settings.DEFAULT_PDF_PATH)}")
        
    uploaded_file = st.file_uploader("Upload de Documento PDF", type=["pdf"])
    
    col1, col2 = st.columns(2)
    with col1:
        btn_index_new = st.button("Processar", type="primary", use_container_width=True, disabled=(uploaded_file is None))
    with col2:
        btn_clear = st.button("Limpar Base", use_container_width=True)

    if btn_index_new and uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
            
        with st.status(f"Indexando {uploaded_file.name}...", expanded=True) as status:
            st.write("Realizando análise estrutural do documento...")
            st.write("Fragmentando parágrafos e gerando embeddings...")
            n = document_processor.index_pdf(tmp_path, force_reindex=False)
            status.update(label="Documento indexado e disponível", state="complete", expanded=False)
            
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
            
        update_stats()
        st.rerun()
        
    if btn_clear:
        with st.spinner("Limpando base vetorial..."):
            if os.path.exists(settings.DEFAULT_PDF_PATH):
                document_processor.index_pdf(settings.DEFAULT_PDF_PATH, force_reindex=True)
            else:
                vector_store.delete_collection()
        update_stats()
        st.rerun()

    st.divider()
    
    # --- PARÂMETROS OPERACIONAIS ---
    st.markdown("#### Parâmetros de Recuperação")
    
    use_reranker = st.toggle("Refinamento Cross-Encoder", value=True)
    
    top_k_retrieval = st.slider("Busca Semântica (Candidatos)", min_value=4, max_value=24, value=settings.TOP_K_RETRIEVAL, step=2)
    
    top_k_rerank = st.slider("Trechos Selecionados", min_value=1, max_value=8, value=settings.TOP_K_RERANK, step=1)
    
    temperature = st.slider("Precisão Técnica (Temperatura)", min_value=0.0, max_value=0.7, value=settings.LLM_TEMPERATURE, step=0.05)

    st.divider()
    st.caption(f"Modelo LLM: {settings.LLM_MODEL_NAME}\n\nServidor: {settings.LLM_BASE_URL}")


# =========================================================
# ÁREA PRINCIPAL - INTERFACE DE CONSULTA
# =========================================================
st.markdown("<h1>Assistente de Procedimentos e Consultas</h1>", unsafe_allow_html=True)
st.markdown("<p style='opacity: 0.75; font-size: 1.05rem; margin-bottom: 2rem;'>Respostas técnicas estruturadas com base em documentação oficial, citações exatas e verificação de fontes.</p>", unsafe_allow_html=True)

if st.session_state.messages:
    col_spacer, col_clear = st.columns([8, 2])
    with col_clear:
        if st.button("Nova Consulta", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and message.get("context_chunks"):
            with st.expander(f"Fontes e Trechos Verificados ({len(message['context_chunks'])})"):
                for idx, chunk in enumerate(message["context_chunks"], 1):
                    page = chunk.get("page_num", "N/A")
                    score = chunk.get("rerank_score", chunk.get("vector_distance", "N/A"))
                    st.markdown(f"""
                    <div class="context-card">
                        <div class="context-header">Trecho {idx} — Página {page} | Índice de Relevância: {score}</div>
                        <div>{chunk['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)

if user_query := st.chat_input("Faça uma pergunta técnica sobre o procedimento ou documento..."):
    if st.session_state.stats["count"] == 0:
        st.error("A base de conhecimento não possui documentos indexados. Utilize o painel lateral para indexar o arquivo padrão ou enviar um PDF.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.status("Consultando documentação e formulando resposta técnica...", expanded=True) as status:
            status.write("Executando busca semântica vetorial (Bi-Encoder)...")
            chunks = retriever.retrieve(
                query=user_query,
                n_retrieval=top_k_retrieval,
                n_rerank=top_k_rerank,
                use_reranker=use_reranker
            )
            
            if use_reranker and len(chunks) > 1:
                status.write("Aplicando refinamento e reordenação com Cross-Encoder...")
            
            status.write("Processando contexto com modelo local...")
            answer = generator.ask(
                question=user_query,
                chunks=chunks,
                temperature=temperature
            )
            status.update(label="Consulta finalizada", state="complete", expanded=False)
        
        st.markdown(answer)
        
        if chunks:
            with st.expander(f"Fontes e Trechos Verificados ({len(chunks)})"):
                for idx, chunk in enumerate(chunks, 1):
                    page = chunk.get("page_num", "N/A")
                    score = chunk.get("rerank_score", chunk.get("vector_distance", "N/A"))
                    st.markdown(f"""
                    <div class="context-card">
                        <div class="context-header">Trecho {idx} — Página {page} | Índice de Relevância: {score}</div>
                        <div>{chunk['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "context_chunks": chunks
    })