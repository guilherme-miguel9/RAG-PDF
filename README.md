# 📖 RAG-PDF — Sistema de Consulta Inteligente a Documentos e POPs

Um sistema de **Retrieval-Augmented Generation (RAG) de nível de produção**, projetado para extrair, processar e responder perguntas sobre Procedimentos Operacionais Padrão (POPs), manuais técnicos e documentos PDF complexos com **alta precisão, citação de páginas e zero alucinação**.

O projeto roda com privacidade **100% local** usando o [LM Studio](https://lmstudio.ai/) e apresenta uma interface interativa e moderna construída com [Streamlit](https://streamlit.io/).

---

## ✨ Destaques e Arquitetura do Sistema

O pipeline foi construído combinando as técnicas mais avançadas do estado da arte em RAG:

```mermaid
graph TD
    A[📄 PDF / POP] -->|Docling| B(Convertido em Markdown Estruturado)
    B -->|Chunking Semântico| C[Trechos + Metadados de Página]
    C -->|Bi-Encoder / mpnet| D[(ChromaDB - Vetores Cosine)]
    
    E[❓ Pergunta do Usuário] -->|Expansão de Query| F[Busca Vetorial - Estágio 1]
    D --> F
    F -->|Top 12 Candidatos| G[🎯 Reranker Cross-Encoder - Estágio 2]
    G -->|Top 4 Trechos Exatos| H[🤖 LLM Local via LM Studio]
    H -->|Resposta + Citação (Página X)| I[🖥️ Interface Streamlit]
```

1. **📄 Ingestão Estruturada com Docling:** Em vez de extração de texto contínuo cru, utilizamos o `Docling` para converter cada página do PDF em **Markdown limpo**, preservando tabelas, listas de verificação, negritos e hierarquias de títulos (com fallback automático para PyMuPDF).
2. **✂️ Chunking Semântico Inteligente:** Divisão de parágrafos e frases que não corta palavras no meio, mantendo sobreposição (`overlap`) coerente e gravando o número exato da página (`page_num`) e hash de deduplicação MD5 para cada trecho.
3. **🔍 Busca Híbrida em Dois Estágios (Retrieval + Reranking):**
   * **Estágio 1 (Busca Vetorial Ampla):** Utiliza o modelo `paraphrase-multilingual-mpnet-base-v2` (ou `multilingual-e5`) para recuperar os 12 candidatos mais relevantes do banco **ChromaDB** com similaridade de cosseno.
   * **Estágio 2 (Cross-Encoder Reranking):** Passa os candidatos recuperados pelo modelo de alta precisão `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, comparando cada trecho com a pergunta original do usuário para selecionar os top 4 trechos cirúrgicos.
4. **🚫 Zero Alucinações e Citação de Páginas:** Prompts rigorosamente desenhados para obrigar o LLM a responder **apenas** com o que está no contexto e citar a página correspondente no formato `(Página X)`.
5. **🔒 Segurança e Privacidade Total:** Nenhum dado sai da sua máquina. O processamento é local via LM Studio. Além disso, regras estritas no `.gitignore` garantem que PDFs, planilhas ou documentos confidenciais nunca sejam acidentalmente commitados no GitHub.

---

## 🛠️ Estrutura do Repositório

```text
RAG-PDF/
├── pdf_chat/
│   ├── app.py              # Interface gráfica principal interativa (Streamlit)
│   ├── config.py           # Central de configurações (caminhos, modelos, portas do LLM)
│   ├── rag_indexer.py      # Módulo de ingestão, conversão Docling e indexação no ChromaDB
│   ├── rag_retriever.py    # Módulo de busca em 2 estágios (Bi-Encoder + Cross-Encoder Reranker)
│   └── rag_llm.py          # Módulo de comunicação com o LLM local e formatação de citações
├── extract_pdf.py          # Script de testes avulso para extração rápida via linha de comando
├── rag_leitura.py          # Script de testes do pipeline básico
├── teste_claude_extract.py # Script de testes do pipeline avançado com reranking
├── requirements.txt        # Dependências Python do projeto
└── README.md               # Documentação oficial
```

---

## 🚀 Como Rodar na Sua Máquina (Passo a Passo)

### 1. Pré-requisitos
* **Python 3.10+** instalado na máquina.
* **Git** instalado.
* **LM Studio** ([download aqui](https://lmstudio.ai/)) instalado para rodar o modelo de inteligência artificial localmente.

### 2. Configurar e Iniciar o LM Studio
1. Abra o **LM Studio** e baixe um modelo recomendável (ex: `meta-llama-3.1-8b-instruct` ou `google/gemma-3-4b`).
2. Acesse a aba **Local Server** (ícone de setas bidirecionais `<->` na barra lateral esquerda).
3. Certifique-se de que a porta está configurada como **`1234`** e clique no botão **Start Server**.

### 3. Clonar o Repositório e Configurar o Ambiente

Abra o terminal (PowerShell, Prompt de Comando ou Terminal do Linux/Mac) e execute os comandos abaixo:

```bash
# 1. Clone o repositório
git clone https://github.com/guilherme-miguel9/RAG-PDF.git
cd RAG-PDF

# 2. Crie um ambiente virtual (recomendado)
python -m venv venv

# 3. Ative o ambiente virtual
# -> No Windows (PowerShell):
.\venv\Scripts\activate
# -> No Linux / macOS:
source venv/bin/activate

# 4. Instale todas as dependências do projeto
pip install -r requirements.txt
```

### 4. Indexar seus Documentos e Abrir o Chat

Você pode indexar o documento diretamente pela interface web do Streamlit:

```bash
# Entre na pasta do aplicativo principal
cd pdf_chat

# Inicie o Streamlit
streamlit run app.py
```

O navegador se abrirá automaticamente na página `http://localhost:8501`.

#### Na Interface Web (`app.py`):
1. **Pela Barra Lateral (`Sidebar`):**
   * Se o seu PDF padrão estiver na raiz do projeto como `pop_leitura.pdf`, clique no botão **`⚡ Indexar POP Padrão`**.
   * Ou, clique em **`📁 Fazer Upload de Novo PDF`** para enviar qualquer documento PDF do seu computador. O sistema converterá via Docling, gerará os blocos semânticos e criará o banco vetorial automaticamente.
2. **Ajuste de Parágrafos em Tempo Real:** Use os sliders da barra lateral para controlar a temperatura do modelo, ativar/desativar o Reranker Cross-Encoder e mudar a quantidade de trechos buscados.
3. **No Chat Principal:** Faça suas perguntas! Abaixo de cada resposta do assistente, clique em **`📚 Ver Trechos e Páginas Consultadas`** para inspecionar os fragmentos exatos do texto que originaram a resposta.

---

## ⚙️ Personalizações (`pdf_chat/config.py`)

Se desejar alterar os modelos de embeddings, a porta do servidor local ou o tamanho dos blocos de texto, basta editar o arquivo `pdf_chat/config.py`:

```python
# Exemplo de configurações disponíveis em config.py:
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
LLM_BASE_URL = "http://127.0.0.1:1234/v1"
LLM_MODEL_NAME = "meta-llama-3.1-8b-instruct"
```

---

## 🔒 Segurança de Dados

Este repositório conta com um `.gitignore` rigorosamente configurado para impedir o envio de planilhas (`.xlsx`, `.csv`), documentos PDF (`.pdf`), chaves de API (`.env`) ou bancos vetoriais (`chroma_db/`) para o GitHub. Todos os seus documentos operacionais permanecem **exclusivamente na sua máquina local**.

---
*Desenvolvido para máxima confiabilidade e precisão técnica em consultas operacionais.*
