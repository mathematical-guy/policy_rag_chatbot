from __future__ import annotations

import groq
import logfire

from rag.config import Settings

SYSTEM_PROMPT = """\
You are an expert insurance policy assistant. Answer the user's question strictly \
based on the provided policy document excerpts.

RULES:
1. Cite every claim with the document reference in the format: \
[doc_id: <doc_id>, page: <page>, section: <section_path>].
2. If the provided context does NOT contain enough information to answer the question, \
respond with: "This is not covered in the provided policy documents. Please consult your \
full policy document or contact your insurer for details."
3. Never hallucinate or infer information not present in the excerpts.
4. Be precise and concise.
5. If there are tables or benefit schedules, reference them specifically.
"""


def generate_answer(
    query: str,
    context_chunks: list[dict],
    cfg: Settings | None = None,
) -> str:
    """
    Generate a grounded, cited answer using Groq LLM.
    """
    if cfg is None:
        from rag.config import get_settings
        cfg = get_settings()

    # Build context string from reranked chunks
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        ref = f"[Chunk {i} | doc: {chunk.get('doc_id', '?')[:12]}… | page: {chunk.get('page', '?')} | section: {chunk.get('section_path', 'N/A')}]"
        context_parts.append(f"{ref}\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    user_message = f"""\
POLICY DOCUMENT EXCERPTS:
{context}

---

USER QUESTION: {query}"""

    client = groq.Groq(api_key=cfg.GROQ_API_KEY)

    with logfire.span("llm_generate", model=cfg.GROQ_MODEL, n_chunks=len(context_chunks)):
        response = client.chat.completions.create(
            model=cfg.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content or ""

    logfire.info(
        "llm_generate_done",
        tokens_in=response.usage.prompt_tokens if response.usage else 0,
        tokens_out=response.usage.completion_tokens if response.usage else 0,
    )
    return answer
