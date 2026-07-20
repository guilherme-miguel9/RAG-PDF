# 📖 RAG-PDF — Sistema de Consulta Inteligente a Documentos e POPs

Um sistema de **Retrieval-Augmented Generation (RAG) de nível de produção**, projetado com uma arquitetura modular profissional para extrair, processar e responder perguntas sobre Procedimentos Operacionais Padrão (POPs), manuais técnicos e documentos PDF complexos com **alta precisão, citação de páginas e zero alucinação**.

O projeto roda com privacidade **100% local** usando o [LM Studio](https://lmstudio.ai/) e apresenta uma interface interativa e moderna construída com [Streamlit](https://streamlit.io/).

---

## ✨ Destaques e Arquitetura do Sistema

O pipeline foi construído combinando as técnicas mais avançadas do estado da arte em RAG:

```mermaid
graph TD
    A[📄 PDF / POP em data/raw] -->|Docling| B(Convertido em Markdown Estruturado)
    B -->|Chunking Semântico| C[Trechos + Metadados de Página]
    C -->|Bi-Encoder / mpnet| D[(ChromaDB em data/chroma_db)]
    
    E[❓ Pergunta do Usuário] -->|Expansão de Query| F[Busca Vetorial - Estágio 1]
    D --> F
    F -->|Top 12 Candidatos| G[🎯 Reranker Cross-Encoder - Estágio 2]
    G -->|Top 4 Trechos Exatos| H[🤖 LLM Local via LM Studio]
    H -->|Resposta + Citação (Página X)| I[🖥️ Interface Streamlit]
```

1. **📄 Ingestão Estruturada com Docling:** Em vez de extração de texto contínuo cru, utilizamos o `Docling` para converter cada página do PDF em **Markdown limpo**, preservando tabelas, listas de verificação, negritos e hierarquias de títulos (com fallback automático para PyMuPDF).
2. **✂️ Chunking Semântico Inteligente (`core/document_processor.py`):** Divisão de parágrafos e frases que não corta palavras no meio, mantendo sobreposição (`overlap`) coerente e gravando o número exato da página (`page_num`) e hash de deduplicação MD5 para cada trecho.
3. **🔍 Busca Híbrida em Dois Estágios (`core/retriever.py`):**
   * **Estágio 1 (Busca Vetorial Ampla):** Utiliza o modelo `paraphrase-multilingual-mpnet-base-v2` (ou `multilingual-e5`) para recuperar os 12 candidatos mais relevantes do banco **ChromaDB** com similaridade de cosseno.
   * **Estágio 2 (Cross-Encoder Reranking):** Passa os candidatos recuperados pelo modelo de alta precisão `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, comparando cada trecho com a pergunta original do usuário para selecionar os top 4 trechos cirúrgicos.
4. **🚫 Zero Alucinações e Citação de Páginas (`core/generator.py`):** Prompts rigorosamente desenhados para obrigar o LLM a responder **apenas** com o que está no contexto e citar a página correspondente no formato `(Página X)`.
5. **🔒 Segurança e Privacidade Total:** Nenhum dado sai da sua máquina. O processamento é local via LM Studio. Além disso, regras estritas no `.gitignore` garantem que PDFs, planilhas ou documentos confidenciais nunca sejam acidentalmente commitados no GitHub.

---

## 🛠️ Arquitetura de Pastas Profissional

O projeto adota uma estrutura modular limpa (Clean Architecture), separando a interface gráfica, configurações, domínio principal (`core/`) e scripts de experimentação:

```text
RAG-PDF/
├── app.py                      # Ponto de entrada principal da Interface Gráfica (Streamlit)
├── requirements.txt            # Dependências Python e pacotes do projeto
├── README.md                   # Documentação oficial
├── config/                     # Configurações globais e prompts
│   ├── __init__.py
│   └── settings.py             # Parâmetros de modelos, portas e caminhos do sistema
├── core/                       # Domínio e regras de negócio do RAG (SOLID)
│   ├── __init__.py
│   ├── document_processor.py   # Extração Docling, conversão Markdown e Chunking Semântico
│   ├── vector_store.py         # Conexão ao ChromaDB, gerenciamento vetorial e estatísticas
│   ├── retriever.py            # Busca em 2 Estágios (Bi-Encoder + Reranker Cross-Encoder)
│   └── generator.py            # Orquestração do LLM local (LM Studio) e formatação de contexto
├── scripts/                    # Scripts de linha de comando, referências e experimentos de P&D
│   ├── cli_indexer.py          # Script CLI de indexação avulsa via linha de comando
│   ├── baseline_rag.py         # Experimento inicial de RAG básico preservado para comparação
│   └── advanced_pipeline_test.py # Script de testes do pipeline avançado na linha de comando
└── data/                       # Armazenamento de dados locais (ignorado no Git por segurança)
    ├── raw/                    # Diretório para PDFs originais (ex: pop_leitura.pdf)
    └── chroma_db/              # Banco de dados vetorial persistente
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

### 4. Abrir o Chat e Indexar Documentos

Com a nova arquitetura, rodar a interface web ficou ainda mais simples (direto da raiz do projeto):

```bash
# Na pasta raiz do projeto (com o venv ativado):
streamlit run app.py
```

O navegador se abrirá automaticamente na página `http://localhost:8501`.

#### Na Interface Web (`app.py`):
1. **Pela Barra Lateral (`Sidebar`):**
   * Se o seu PDF padrão estiver na pasta `data/raw/pop_leitura.pdf`, clique no botão **`⚡ Indexar POP Padrão`**.
   * Ou, clique em **`📁 Fazer Upload de Novo PDF`** para enviar qualquer documento PDF do seu computador. O sistema converterá via Docling, gerará os blocos semânticos e criará o banco vetorial automaticamente.
2. **Ajuste de Parágrafos em Tempo Real:** Use os sliders da barra lateral para controlar a temperatura do modelo, ativar/desativar o Reranker Cross-Encoder e mudar a quantidade de trechos buscados.
3. **No Chat Principal:** Faça suas perguntas! Abaixo de cada resposta do assistente, clique em **`📚 Ver Trechos e Páginas Consultadas`** para inspecionar os fragmentos exatos do texto que originaram a resposta.

---

## ⚙️ Personalizações (`config/settings.py`)

Se desejar alterar os modelos de embeddings, a porta do servidor local ou o tamanho dos blocos de texto, basta editar o arquivo centralizado `config/settings.py`:

```python
# Exemplo de configurações disponíveis em config/settings.py:
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
LLM_BASE_URL = "http://127.0.0.1:1234/v1"
LLM_MODEL_NAME = "meta-llama-3.1-8b-instruct"
```

---

## 🔒 Segurança de Dados

Este repositório conta com um `.gitignore` rigorosamente configurado para impedir o envio de planilhas (`.xlsx`, `.csv`), documentos PDF (`.pdf`), chaves de API (`.env`) ou pastas inteiras do banco vetorial (`data/`) para o GitHub. Todos os seus documentos operacionais permanecem **exclusivamente na sua máquina local**.

---
*Desenvolvido seguindo os mais altos padrões de arquitetura limpa, modularidade e precisão técnica.*
