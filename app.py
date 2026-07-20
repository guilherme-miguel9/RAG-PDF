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
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# =========================================================
st.set_page_config(
    page_title="POP Intelligence — LangChain",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# DESIGN SYSTEM PROFISSIONAL — APPLE PRO DARK MODE UNIFICADO
# =========================================================
APPLE_PRO_DARK_CSS = """
<style>
/* Importação de tipografia limpa estilo Apple */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Ocultar elementos nativos de cabeçalho e rodapé do Streamlit */
header { visibility: hidden; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

/* Configuração de fundo escuro e fontes unificadas */
html, body, .stApp, div[data-testid="stAppViewContainer"], div[data-testid="stMain"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", sans-serif !important;
    color: #F5F5F7 !important;
    background-color: #0A0A0C !important;
    -webkit-font-smoothing: antialiased;
}

/* Container principal ajustado para espaçamento generoso e centralizado */
.block-container {
    max-width: 1140px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 4rem !important;
}

/* Forçar cor de todos os títulos nativos e customizados para Branco Puro (#FFFFFF) */
h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
}

/* Textos descritivos e parágrafos */
p, span, label, .stMarkdown p {
    color: #A1A1A6;
}

/* Barra de Navegação Superior Estilo Apple Pro Dark */
.apple-navbar {
    width: 100%;
    background: rgba(28, 28, 30, 0.85);
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 0.85rem 1.5rem;
    margin-bottom: 2.5rem;
    border-radius: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.apple-nav-brand {
    font-weight: 600;
    font-size: 1.05rem;
    letter-spacing: -0.02em;
    color: #FFFFFF !important;
}
.apple-nav-links {
    display: flex;
    gap: 1.8rem;
    font-size: 0.88rem;
    font-weight: 400;
}
.apple-nav-links span {
    color: #A1A1A6 !important;
    transition: color 0.2s ease;
}
.apple-nav-links span:hover {
    color: #FFFFFF !important;
}

/* Seção Hero Estilo Apple Store Pro */
.apple-hero {
    text-align: center;
    padding: 3rem 1rem 3.5rem 1rem;
    max-width: 800px;
    margin: 0 auto;
}
.apple-pill-badge {
    display: inline-block;
    background-color: #1C1C1E;
    color: #0A84FF !important;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.35rem 0.9rem;
    border-radius: 9999px;
    border: 1px solid rgba(10, 132, 255, 0.35);
    margin-bottom: 1.2rem;
    letter-spacing: -0.01em;
}
.apple-title {
    font-size: 3.4rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.035em !important;
    line-height: 1.08 !important;
    color: #FFFFFF !important;
    margin-bottom: 1rem !important;
}
.apple-subtitle {
    font-size: 1.25rem !important;
    font-weight: 400 !important;
    color: #A1A1A6 !important;
    line-height: 1.45 !important;
    letter-spacing: -0.015em !important;
}

/* Cartões e Contêineres Apple Pro Dark (Superfície Translúcida) */
.apple-card {
    background: #1C1C1E !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 18px;
    padding: 1.8rem;
    box-shadow: 0 12px 34px rgba(0, 0, 0, 0.5);
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
}
.apple-card:hover {
    border-color: rgba(255, 255, 255, 0.22) !important;
    box-shadow: 0 16px 42px rgba(0, 0, 0, 0.65);
}
.apple-card-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #FFFFFF !important;
    margin-bottom: 0.4rem;
    letter-spacing: -0.02em;
}
.apple-card-desc {
    font-size: 0.92rem;
    color: #A1A1A6 !important;
    line-height: 1.45;
}

/* Customização de Botões - Formato Pílula Apple Pro */
div.stButton > button {
    border-radius: 9999px !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    padding: 0.55rem 1.4rem !important;
    border: none !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
}
/* Botão primário azul Apple (#0A84FF) com texto branco */
div.stButton > button:first-child {
    background-color: #0A84FF !important;
    color: #FFFFFF !important;
}
div.stButton > button:first-child:hover {
    background-color: #0071E3 !important;
    transform: scale(1.015);
    box-shadow: 0 4px 14px rgba(10, 132, 255, 0.4) !important;
}

/* Caixa de Upload de Arquivo */
div[data-testid="stFileUploader"] section {
    border-radius: 16px !important;
    border: 1.5px dashed rgba(255, 255, 255, 0.25) !important;
    background-color: #1C1C1E !important;
    padding: 1.5rem !important;
}
div[data-testid="stFileUploader"] section span, div[data-testid="stFileUploader"] section small {
    color: #A1A1A6 !important;
}

/* Ajustes na Barra Lateral (Sidebar Apple Pro Dark) */
section[data-testid="stSidebar"] {
    background-color: #161618 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem !important;
}
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

/* Campos de entrada, sliders e caixas de texto */
.stTextInput > div > div > input, .stTextArea > div > div > textarea, div[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    padding: 0.65rem 0.9rem !important;
    font-size: 0.95rem !important;
    background-color: #1C1C1E !important;
    color: #FFFFFF !important;
}
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus, div[data-testid="stChatInput"] textarea:focus {
    border-color: #0A84FF !important;
    box-shadow: 0 0 0 3px rgba(10, 132, 255, 0.25) !important;
}

/* Caixas de resposta e mensagens do chat */
[data-testid="stChatMessage"] {
    background-color: #161618 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    padding: 1.25rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
    color: #F5F5F7 !important;
}

/* Expansores (Expanders) */
div[data-testid="stExpander"] {
    background-color: #1C1C1E !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] p {
    color: #A1A1A6 !important;
}
</style>
"""
st.markdown(APPLE_PRO_DARK_CSS, unsafe_allow_html=True)


# =========================================================
# BARRA DE NAVEGAÇÃO SUPERIOR ESTILO APPLE PRO DARK
# =========================================================
st.markdown("""
<div class="apple-navbar">
    <div class="apple-nav-brand">LangChain RAG Architecture</div>
    <div class="apple-nav-links">
        <span>Procedimentos Operacionais</span>
        <span>Cross-Encoder Rerank</span>
        <span>Inferência Local</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SEÇÃO HERO — APRESENTAÇÃO APPLE PRO
# =========================================================
st.markdown("""
<div class="apple-hero">
    <div class="apple-pill-badge">Motor LangChain & Docling Integrados</div>
    <h1 class="apple-title">Consulta Operacional.<br>Sintetizada com precisão.</h1>
    <p class="apple-subtitle">Sistema de inteligência documental para análise de Procedimentos Operacionais Padrão com verificação cruzada de evidências.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# BARRA LATERAL (SIDEBAR) — CONTROLE DE PARÂMETROS
# =========================================================
with st.sidebar:
    st.markdown("<h3 style='font-weight:600; font-size:1.1rem; margin-bottom:1rem; color:#FFFFFF;'>Estado da Base Vetorial</h3>", unsafe_allow_html=True)
    
    stats = vector_store.get_collection_stats()
    if stats["exists"] and stats["count"] > 0:
        st.markdown(f"""
        <div style="background:#1C1C1E; padding:1.1rem; border-radius:14px; border:1px solid rgba(255,255,255,0.12); margin-bottom:1.2rem;">
            <div style="font-size:0.8rem; color:#A1A1A6; text-transform:uppercase; letter-spacing:0.04em; font-weight:500;">Documentos Indexados</div>
            <div style="font-size:2rem; font-weight:700; color:#FFFFFF; margin-top:0.2rem;">{stats["count"]}</div>
            <div style="font-size:0.8rem; color:#0A84FF; margin-top:0.3rem; font-weight:500;">Coleção LangChain ativa</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#1C1C1E; padding:1.1rem; border-radius:14px; border:1px solid rgba(255,255,255,0.12); margin-bottom:1.2rem;">
            <div style="font-size:0.8rem; color:#A1A1A6; text-transform:uppercase; letter-spacing:0.04em; font-weight:500;">Estado do Banco</div>
            <div style="font-size:1.1rem; font-weight:600; color:#A1A1A6; margin-top:0.2rem;">Nenhum documento</div>
            <div style="font-size:0.8rem; color:#A1A1A6; margin-top:0.3rem;">Realize a carga na área principal</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.1); margin:1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-weight:600; font-size:1.1rem; margin-bottom:1rem; color:#FFFFFF;'>Parâmetros de Operação</h3>", unsafe_allow_html=True)
    
    top_k_retrieval = st.slider("Recuperação Inicial (Bi-Encoder)", min_value=3, max_value=20, value=settings.TOP_K_RETRIEVAL, help="Quantidade de fragmentos extraídos na primeira etapa via similaridade de cosseno.")
    top_k_rerank = st.slider("Seleção Final (Cross-Encoder)", min_value=1, max_value=8, value=settings.TOP_K_RERANK, help="Quantidade de fragmentos refinados e injetados na cadeia de inferência.")
    
    st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.1); margin:1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-weight:600; font-size:1.1rem; margin-bottom:1rem; color:#FFFFFF;'>Conexão de Inferência</h3>", unsafe_allow_html=True)
    st.caption(f"Servidor: `{settings.LLM_BASE_URL}`")
    st.caption(f"Modelo: `{settings.LLM_MODEL_NAME}`")
    
    if st.button("Limpar Base Vetorial LangChain", use_container_width=True):
        vector_store.delete_collection()
        st.success("Base vetorial removida com sucesso.")
        st.rerun()


# =========================================================
# GESTÃO DE DOCUMENTOS E CARGA NA BASE
# =========================================================
st.markdown("<div style='margin-bottom:1.8rem;'></div>", unsafe_allow_html=True)

col_doc1, col_doc2 = st.columns([1.3, 1], gap="large")

with col_doc1:
    st.markdown("""
    <div class="apple-card">
        <div class="apple-card-title">Carga e Indexação de Documentos</div>
        <div class="apple-card-desc">Envie um novo arquivo PDF do Procedimento Operacional Padrão ou utilize o documento local residente no servidor. O motor Docling estruturará tabelas e parágrafos antes da conversão em objetos LangChain.</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Selecione o arquivo PDF para processamento", type=["pdf"], label_visibility="collapsed")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if uploaded_file is not None:
            if st.button("Indexar Arquivo Selecionado", use_container_width=True):
                with st.status("Processando documento e indexando via LangChain...", expanded=True) as status:
                    st.write("Gravando temporariamente no disco...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                        
                    st.write("Executando conversão estruturada via Docling e fragmentação semântica...")
                    count = document_processor.index_pdf(tmp_path, force_reindex=False)
                    os.unlink(tmp_path)
                    
                    status.update(label=f"Indexação concluída com sucesso. Total de fragmentos: {count}", state="complete", expanded=False)
                st.rerun()
        else:
            st.button("Indexar Arquivo Selecionado", disabled=True, use_container_width=True)
            
    with col_btn2:
        default_pdf = settings.DEFAULT_PDF_PATH
        if os.path.exists(default_pdf):
            if st.button("Indexar Documento Local (pop_leitura.pdf)", use_container_width=True):
                with st.status("Carregando e indexando documento padrão via LangChain...", expanded=True) as status:
                    st.write("Analisando estrutura documental do PDF residente...")
                    count = document_processor.index_pdf(default_pdf, force_reindex=False)
                    status.update(label=f"Documento local indexado com sucesso. Fragmentos na coleção: {count}", state="complete", expanded=False)
                st.rerun()

with col_doc2:
    st.markdown("""
    <div class="apple-card" style="height: 100%;">
        <div class="apple-card-title">Arquitetura LangChain LCEL</div>
        <div class="apple-card-desc" style="margin-top:0.8rem;">
            O fluxo de processamento opera de maneira encadeada:
            <ul style="margin-top:0.6rem; padding-left:1.2rem; line-height:1.6; color:#A1A1A6;">
                <li><b style="color:#FFFFFF;">Ingestão:</b> Conversão de layout Markdown via Docling para objetos <code>Document</code> oficiais do LangChain.</li>
                <li><b style="color:#FFFFFF;">Armazenamento:</b> Vetorização local de alta densidade no Chroma via <code>HuggingFaceEmbeddings</code>.</li>
                <li><b style="color:#FFFFFF;">Retriever Duplo:</b> Busca inicial por cosseno e reordenação de precisão via <code>CrossEncoder</code>.</li>
                <li><b style="color:#FFFFFF;">Cadeia LCEL:</b> Formatação e geração limpa via <code>ChatPromptTemplate</code> conectado ao LM Studio.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# INTERFACE DE CONSULTA E RESPOSTAS (CHAT APPLE STYLE)
# =========================================================
st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.1); margin:2.5rem 0;'>", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <h2 style="font-size:1.8rem; font-weight:700; letter-spacing:-0.025em; color:#FFFFFF; margin-bottom:0.3rem;">Consulta ao Procedimento Operacional</h2>
    <p style="font-size:1rem; color:#A1A1A6;">Digite sua dúvida técnica abaixo para receber uma síntese baseada estritamente no texto do documento.</p>
</div>
""", unsafe_allow_html=True)

# Histórico de conversação na sessão do Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens anteriores
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Evidências Documentais Recuperadas"):
                for src in msg["sources"]:
                    st.markdown(f"**Página {src['page']} (Relevância Cross-Encoder: {src['score']:.4f})**\n\n{src['text']}\n\n---")

# Entrada do usuário
query = st.chat_input("Ex: Quais os requisitos de segurança para inspeção ou qual o procedimento para falhas?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    with st.chat_message("assistant"):
        with st.spinner("Analisando evidências na base vetorial e gerando síntese técnica..."):
            retrieved = retriever.retrieve(query, top_k_retrieval=top_k_retrieval, top_k_rerank=top_k_rerank)
            
            if not retrieved:
                answer = "Com base no procedimento operacional indexado no momento, não foram encontrados trechos com similaridade suficiente para responder a esta consulta com precisão."
                sources_meta = []
            else:
                answer = generator.generate_answer(query, retrieved)
                sources_meta = [
                    {
                        "page": c.get("page_num", "?"),
                        "score": c.get("rerank_score", c.get("similarity", 0.0)),
                        "text": c.get("text", "")
                    }
                    for c in retrieved
                ]
                
            st.markdown(answer)
            if sources_meta:
                with st.expander("Evidências Documentais Recuperadas"):
                    for src in sources_meta:
                        st.markdown(f"**Página {src['page']} (Relevância Cross-Encoder: {src['score']:.4f})**\n\n{src['text']}\n\n---")
                        
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources_meta
    })