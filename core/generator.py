from openai import OpenAI
from config import settings

# Inicializa o cliente apontando para a base URL do LM Studio local
client = OpenAI(
    base_url=settings.LLM_BASE_URL,
    api_key=settings.LLM_API_KEY
)


def format_context(chunks: list[dict]) -> str:
    """Formata os trechos recuperados para inclusão no prompt do LLM indicando páginas."""
    if not chunks:
        return "Nenhum trecho de contexto relevante foi recuperado."
        
    parts = []
    for c in chunks:
        page = c.get("page_num", "N/A")
        score = c.get("rerank_score", c.get("vector_distance", "N/A"))
        text = c.get("text", "")
        
        parts.append(f"[Página {page} | Relevância: {score}]\n{text}")
        
    return "\n\n---\n\n".join(parts)


def ask(
    question: str,
    chunks: list[dict],
    temperature: float = settings.LLM_TEMPERATURE,
    max_tokens: int = settings.LLM_MAX_TOKENS
) -> str:
    """
    Envia o prompt com o contexto recuperado e a pergunta para o LLM local.
    Retorna a resposta com fidelidade estrita às fontes.
    """
    if not chunks:
        return (
            "⚠️ Não encontrei informações suficientes no documento indexado para responder "
            "com segurança a essa pergunta."
        )

    context_str = format_context(chunks)
    
    user_message = f"""CONTEXTO RECUPERADO DO DOCUMENTO:
{context_str}

===
PERGUNTA DO USUÁRIO:
{question}

RESPONDA UTILIZANDO APENAS O CONTEXTO ACIMA. LEMBRE-SE DE CITAR A PÁGINA AO FINAL DAS INFORMAÇÕES."""

    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": settings.SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        error_msg = str(e)
        if "Connection refused" in error_msg or "127.0.0.1:1234" in error_msg:
            return (
                "❌ **Erro de Conexão com o LM Studio:** Não foi possível conectar ao servidor local `http://127.0.0.1:1234/v1`.\n\n"
                "👉 *Por favor, verifique se o LM Studio está aberto, se o servidor local (Local Server) foi iniciado na porta 1234 e se o modelo está carregado.*"
            )
        return f"❌ **Erro ao gerar resposta com o LLM:** {error_msg}"