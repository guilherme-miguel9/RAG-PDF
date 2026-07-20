from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import settings

# Inicializa o modelo de chat do LangChain apontando para o servidor local LM Studio
llm = ChatOpenAI(
    base_url=settings.LLM_BASE_URL,
    api_key=settings.LLM_API_KEY,
    model=settings.LLM_MODEL_NAME,
    temperature=settings.LLM_TEMPERATURE
)

SYSTEM_PROMPT = """Você é o Assistente Especialista em Procedimentos Operacionais Padrão (POP), operando com motor LangChain RAG.
Sua missão é fornecer respostas técnicas, exatas, profissionais e rigorosamente embasadas nos trechos do documento fornecidos.

Regras Inegociáveis de Atuação:
1. Baseie sua resposta EXCLUSIVAMENTE nas informações contidas nos trechos de contexto abaixo.
2. Nunca invente, presuma ou extrapole procedimentos não descritos no texto ("alucinação zero").
3. Cite sempre o número da página correspondente ao fato ou procedimento mencionado (ex: [Página 4]).
4. Se a informação solicitada não estiver presente no contexto fornecido, responda com clareza profissional:
   "Com base no procedimento operacional indexado no momento, não há menção específica sobre este ponto."
5. Mantenha um tom sóbrio, analítico, estruturado e altamente claro. Utilize listas numeradas ou tópicos se facilitar a leitura técnica."""


def format_context(chunks: list[dict]) -> str:
    """Formata os trechos recuperados para injeção no prompt do LangChain, com referências claras."""
    if not chunks:
        return "Nenhum trecho relevante identificado para esta consulta."
        
    formatted_parts = []
    for i, c in enumerate(chunks, 1):
        text = c.get("text", "").strip()
        page = c.get("page_num", "?")
        score = c.get("rerank_score", c.get("similarity", 0.0))
        formatted_parts.append(f"--- TRECHO {i} [Página {page} | Relevância: {score:.4f}] ---\n{text}")
        
    return "\n\n".join(formatted_parts)


def generate_answer(query: str, retrieved_chunks: list[dict], custom_system_prompt: str = None) -> str:
    """
    Gera a resposta técnica utilizando a arquitetura LangChain LCEL (Prompt -> LLM -> OutputParser).
    """
    context_text = format_context(retrieved_chunks)
    active_system_prompt = custom_system_prompt if custom_system_prompt else SYSTEM_PROMPT
    
    # Criação do template LangChain
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", active_system_prompt),
        ("user", "Contexto Documental Oficial:\n{context}\n\nPergunte do Operador/Usuário:\n{query}")
    ])
    
    # Cadeia LCEL (LangChain Expression Language)
    rag_chain = prompt_template | llm | StrOutputParser()
    
    print(f"[LangChain Generator] Acionando cadeia LCEL para o modelo '{settings.LLM_MODEL_NAME}' em {settings.LLM_BASE_URL}...")
    try:
        response = rag_chain.invoke({
            "context": context_text,
            "query": query
        })
        return response.strip()
    except Exception as e:
        error_msg = f"[Erro na Cadeia LangChain]: Falha ao comunicar com o servidor LM Studio em '{settings.LLM_BASE_URL}'.\nDetalhes do Erro: {str(e)}"
        print(error_msg)
        return "Não foi possível obter resposta do servidor de inferência local (LM Studio). Verifique se o servidor local está em execução e acessível."