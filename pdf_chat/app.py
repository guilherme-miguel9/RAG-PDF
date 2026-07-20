import os
import tempfile
import streamlit as st

import config
from rag_indexer import index_pdf, get_collection_stats
from rag_retriever import retrieve
from rag_llm import ask

# =========================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# =========================================================
st.set_page_config(
    page_title="RAG Leitura - Assistente de POPs e PDFs",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS adicional para elegância
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
# INICIALIZAÇÃO DE ESTADO DA SESSÃO
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "stats" not in st.session_state:
    st.session_state.stats = get_collection_stats()


def update_stats():
    """Atualiza as estatísticas do banco vetorial no estado da sessão."""
    st.session_state.stats = get_collection_stats()


# =========================================================
# BARRA LATERAL (SIDEBAR - CONTROLES E CONFIGURAÇÕES)
# =========================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/pdf-2--v1.png", width=64)
    st.title("⚙️ Painel do RAG")
    st.caption("Sistema Avançado com Docling, Bi-Encoder e Reranking Cross-Encoder")
    
    st.divider()
    
    # --- ESTATÍSTICAS DO BANCO VETORIAL ---
    st.subheader("📊 Base Vetorial (ChromaDB)")
    stats = st.session_state.stats
    if stats["count"] > 0:
        st.success(f"🟢 **Índice Ativo:** `{stats['collection_name']}`")
        st.metric("Trechos Indexados (Chunks)", stats["count"])
    else:
        st.warning("⚠️ **Nenhum documento indexado ainda.**")
        st.metric("Trechos Indexados", 0)
        
    st.divider()
    
    # --- INGESTÃO DE DOCUMENTOS ---
    st.subheader("📥 Indexação de Documentos")
    
    # Botão rápido para indexar o PDF padrão do projeto (pop_leitura.pdf)
    if os.path.exists(config.DEFAULT_PDF_PATH):
        if st.button("⚡ Indexar POP Padrão (`pop_leitura.pdf`)", use_container_width=True):
            with st.status("Rotina de indexação iniciada...", expanded=True) as status:
                st.write("Extraindo páginas e dividindo em blocos semânticos...")
                n = index_pdf(config.DEFAULT_PDF_PATH, force_reindex=False)
                status.update(label="✅ Indexação Concluída com Sucesso!", state="complete", expanded=False)
            update_stats()
            st.toast(f"🎉 {n} trechos prontos para consulta!", icon="🚀")
            st.rerun()
    else:
        st.info(f"💡 PDF padrão não encontrado em: `{os.path.basename(config.DEFAULT_PDF_PATH)}`")
        
    # Upload de novo PDF
    uploaded_file = st.file_uploader("📁 Fazer Upload de Novo PDF", type=["pdf"])
    
    col1, col2 = st.columns(2)
    with col1:
        btn_index_new = st.button("🚀 Indexar Upload", type="primary", use_container_width=True, disabled=(uploaded_file is None))
    with col2:
        btn_clear = st.button("🗑️ Recriar Base", use_container_width=True, help="Apaga todos os chunks e zera o banco")

    if btn_index_new and uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
            
        with st.status(f"Processando `{uploaded_file.name}`...", expanded=True) as status:
            st.write("📄 Convertendo páginas com Docling...")
            st.write("✂️ Gerando chunks semânticos...")
            st.write("🧠 Criando embeddings e salvando no ChromaDB...")
            n = index_pdf(tmp_path, force_reindex=False)
            status.update(label=f"✅ `{uploaded_file.name}` indexado! ({n} trechos)", state="complete", expanded=False)
            
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
            
        update_stats()
        st.toast("🎉 Arquivo indexado e pronto!", icon="📚")
        st.rerun()
        
    if btn_clear:
        with st.spinner("Apagando e recriando base vetorial..."):
            if os.path.exists(config.DEFAULT_PDF_PATH):
                index_pdf(config.DEFAULT_PDF_PATH, force_reindex=True)
            else:
                # Cria coleção vazia forçadamente
                client = index_pdf.__globals__["client"]
                try:
                    client.delete_collection(config.COLLECTION_NAME)
                except Exception:
                    pass
        update_stats()
        st.toast("🧹 Base vetorial recriada!", icon="✨")
        st.rerun()

    st.divider()
    
    # --- PARÂMETROS DE BUSCA E LLM ---
    st.subheader("🎛️ Parâmetros em Tempo Real")
    
    use_reranker = st.toggle("🎯 Ativar Reranker (Cross-Encoder)", value=True, help="Refina os trechos usando o modelo mmarco-mMiniLMv2 para máxima precisão.")
    
    top_k_retrieval = st.slider("🔍 Top-K Busca Vetorial (Estágio 1)", min_value=4, max_value=24, value=config.TOP_K_RETRIEVAL, step=2)
    
    top_k_rerank = st.slider("📑 Top-K Trechos Finais (Estágio 2)", min_value=1, max_value=8, value=config.TOP_K_RERANK, step=1)
    
    temperature = st.slider("🌡️ Temperatura do LLM", min_value=0.0, max_value=0.7, value=config.LLM_TEMPERATURE, step=0.05, help="Valores menores = respostas mais estritamente fiéis ao texto.")

    st.divider()
    st.caption(f"🤖 **Modelo:** `{config.LLM_MODEL_NAME}`\n\n🔗 **Servidor:** `{config.LLM_BASE_URL}`")


# =========================================================
# ÁREA PRINCIPAL - CHAT INTERATIVO
# =========================================================
st.title("📖 Assistente Inteligente de Leitura e POPs")
st.markdown("Fale diretamente com os seus documentos. Respostas estritamente baseadas no texto, com citações automáticas de página e sem alucinações.")

# Botão para limpar histórico do chat na tela
if st.session_state.messages:
    if st.button("🧹 Limpar Histórico do Chat"):
        st.session_state.messages = []
        st.rerun()

# Exibe histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Se for mensagem do assistente e tiver trechos de contexto associados
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

# Entrada de pergunta do usuário
if user_query := st.chat_input("Ex: Quais são as regras para leitura do medidor no POP? Como proceder em caso de erro?"):
    # Verifica se há algo no banco vetorial
    if st.session_state.stats["count"] == 0:
        st.error("⚠️ O banco vetorial está vazio! Por favor, clique em **Indexar POP Padrão** ou faça o upload de um PDF na barra lateral antes de perguntar.")
        st.stop()

    # Adiciona pergunta ao histórico e exibe
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Gera resposta do assistente
    with st.chat_message("assistant"):
        with st.status("🧠 Consultando base de conhecimento e raciocinando...", expanded=True) as status:
            status.write("🔍 [Estágio 1] Realizando busca semântica vetorial (Bi-Encoder)...")
            chunks = retrieve(
                query=user_query,
                n_retrieval=top_k_retrieval,
                n_rerank=top_k_rerank,
                use_reranker=use_reranker
            )
            
            if use_reranker and len(chunks) > 1:
                status.write(f"🎯 [Estágio 2] Aplicando Reranking de alta precisão nos top {len(chunks)} trechos...")
            
            status.write("🤖 Consultando LLM no LM Studio e estruturando resposta...")
            answer = ask(
                question=user_query,
                chunks=chunks,
                temperature=temperature
            )
            status.update(label="✅ Resposta Gerada!", state="complete", expanded=False)
        
        # Exibe a resposta final na tela
        st.markdown(answer)
        
        # Exibe o Expander dos trechos recuperados logo abaixo
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
                    
    # Salva no histórico com o contexto
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "context_chunks": chunks
    })