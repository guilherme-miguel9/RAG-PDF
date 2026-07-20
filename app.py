import os
import tempfile
import streamlit as st

from config import settings
from core import document_processor, vector_store, retriever, generator

# =========================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# =========================================================
st.set_page_config(
    page_title="RAG-PDF | Assistente Inteligente de POPs",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stChatInput {
        padding-bottom: 20px;
    }
    .chunk-box {
        background-color: rgba(128, 128, 128, 0.1);
        border-left: 4px solid #4CAF50;
        padding: 12px;
        margin-bottom: 12px;
        border-radius: 4px;
        font-size: 0.9em;
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
# BARRA LATERAL (SIDEBAR - CONTROLES)
# =========================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/pdf-2--v1.png", width=64)
    st.title("⚙️ Painel do RAG")
    st.caption("Sistema Arquitetural com Docling, Bi-Encoder e Rerank Cross-Encoder")
    
    st.divider()
    
    # --- ESTATÍSTICAS DA COLEÇÃO ---
    st.subheader("📊 Base Vetorial (ChromaDB)")
    stats = st.session_state.stats
    if stats["count"] > 0:
        st.success(f"🟢 **Índice Ativo:** `{stats['collection_name']}`")
        st.metric("Trechos Indexados (Chunks)", stats["count"])
    else:
        st.warning("⚠️ **Nenhum documento indexado no momento.**")
        st.metric("Trechos Indexados", 0)
        
    st.divider()
    
    # --- INGESTÃO E PROCESSAMENTO DE DOCUMENTOS ---
    st.subheader("📥 Ingestão de Documentos")
    
    if os.path.exists(settings.DEFAULT_PDF_PATH):
        if st.button("⚡ Indexar POP Padrão (`pop_leitura.pdf`)", use_container_width=True):
            with st.status("Rotina de indexação iniciada...", expanded=True) as status:
                st.write("📄 Convertendo páginas para Markdown (Docling)...")
                st.write("✂️ Dividindo em blocos semânticos...")
                n = document_processor.index_pdf(settings.DEFAULT_PDF_PATH, force_reindex=False)
                status.update(label="✅ Indexação Concluída com Sucesso!", state="complete", expanded=False)
            update_stats()
            st.toast(f"🎉 {n} trechos prontos para consulta!", icon="🚀")
            st.rerun()
    else:
        st.info(f"💡 PDF padrão não encontrado em: `data/raw/{os.path.basename(settings.DEFAULT_PDF_PATH)}`")
        
    uploaded_file = st.file_uploader("📁 Fazer Upload de Novo PDF", type=["pdf"])
    
    col1, col2 = st.columns(2)
    with col1:
        btn_index_new = st.button("🚀 Indexar Upload", type="primary", use_container_width=True, disabled=(uploaded_file is None))
    with col2:
        btn_clear = st.button("🗑️ Recriar Base", use_container_width=True, help="Remove a coleção e limpa todos os vetores do banco")

    if btn_index_new and uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
            
        with st.status(f"Processando `{uploaded_file.name}`...", expanded=True) as status:
            st.write("📄 Convertendo documento via Docling...")
            st.write("✂️ Aplicando chunking semântico inteligente...")
            st.write("🧠 Gerando embeddings no ChromaDB...")
            n = document_processor.index_pdf(tmp_path, force_reindex=False)
            status.update(label=f"✅ `{uploaded_file.name}` indexado! ({n} trechos)", state="complete", expanded=False)
            
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
            
        update_stats()
        st.toast("🎉 Arquivo indexado e disponível para o assistente!", icon="📚")
        st.rerun()
        
    if btn_clear:
        with st.spinner("Apagando e recriando base vetorial..."):
            if os.path.exists(settings.DEFAULT_PDF_PATH):
                document_processor.index_pdf(settings.DEFAULT_PDF_PATH, force_reindex=True)
            else:
                vector_store.delete_collection()
        update_stats()
        st.toast("🧹 Base vetorial recriada com sucesso!", icon="✨")
        st.rerun()

    st.divider()
    
    # --- PARÂMETROS OPERACIONAIS ---
    st.subheader("🎛️ Parâmetros de Busca e Gerador")
    
    use_reranker = st.toggle("🎯 Ativar Reranker (Cross-Encoder)", value=True, help="Refina os trechos usando o modelo mmarco-mMiniLMv2 para máxima exatidão na resposta.")
    
    top_k_retrieval = st.slider("🔍 Top-K Busca Vetorial (Estágio 1)", min_value=4, max_value=24, value=settings.TOP_K_RETRIEVAL, step=2)
    
    top_k_rerank = st.slider("📑 Top-K Trechos Finais (Estágio 2)", min_value=1, max_value=8, value=settings.TOP_K_RERANK, step=1)
    
    temperature = st.slider("🌡️ Temperatura do LLM", min_value=0.0, max_value=0.7, value=settings.LLM_TEMPERATURE, step=0.05, help="Valores menores = fidelidade estrita às fontes sem criatividade externa.")

    st.divider()
    st.caption(f"🤖 **Modelo:** `{settings.LLM_MODEL_NAME}`\n\n🔗 **Servidor:** `{settings.LLM_BASE_URL}`")


# =========================================================
# ÁREA PRINCIPAL - CHAT INTERATIVO
# =========================================================
st.title("📖 RAG-PDF | Assistente de Consultas e POPs")
st.markdown("Consulte procedimentos operacionais e documentos com respostas precisas baseadas exclusivamente no texto indexado e citações automáticas de páginas.")

if st.session_state.messages:
    if st.button("🧹 Limpar Histórico do Chat"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and message.get("context_chunks"):
            with st.expander(f"📚 Ver {len(message['context_chunks'])} Trecho(s) de Contexto Consultados"):
                for idx, chunk in enumerate(message["context_chunks"], 1):
                    page = chunk.get("page_num", "N/A")
                    score = chunk.get("rerank_score", chunk.get("vector_distance", "N/A"))
                    st.markdown(f"""
                    <div class="chunk-box">
                        <strong>📌 Trecho #{idx} — Página {page}</strong> (Relevância: <code>{score}</code>)<br><br>
                        {chunk['text']}
                    </div>
                    """, unsafe_allow_html=True)

if user_query := st.chat_input("Ex: Quais são as regras para leitura do medidor? Como proceder em caso de erro no POP?"):
    if st.session_state.stats["count"] == 0:
        st.error("⚠️ O banco vetorial está vazio! Por favor, clique em **Indexar POP Padrão** ou faça o upload de um PDF na barra lateral antes de perguntar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.status("🧠 Consultando base de conhecimento e raciocinando...", expanded=True) as status:
            status.write("🔍 [Estágio 1] Realizando busca semântica vetorial (Bi-Encoder)...")
            chunks = retriever.retrieve(
                query=user_query,
                n_retrieval=top_k_retrieval,
                n_rerank=top_k_rerank,
                use_reranker=use_reranker
            )
            
            if use_reranker and len(chunks) > 1:
                status.write(f"🎯 [Estágio 2] Aplicando Reranking de precisão nos top {len(chunks)} candidatos...")
            
            status.write("🤖 Consultando LLM no LM Studio e formulando resposta técnica...")
            answer = generator.ask(
                question=user_query,
                chunks=chunks,
                temperature=temperature
            )
            status.update(label="✅ Resposta Gerada!", state="complete", expanded=False)
        
        st.markdown(answer)
        
        if chunks:
            with st.expander(f"📚 Ver {len(chunks)} Trecho(s) de Contexto Consultados"):
                for idx, chunk in enumerate(chunks, 1):
                    page = chunk.get("page_num", "N/A")
                    score = chunk.get("rerank_score", chunk.get("vector_distance", "N/A"))
                    st.markdown(f"""
                    <div class="chunk-box">
                        <strong>📌 Trecho #{idx} — Página {page}</strong> (Relevância: <code>{score}</code>)<br><br>
                        {chunk['text']}
                    </div>
                    """, unsafe_allow_html=True)
                    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "context_chunks": chunks
    })