import requests
from config import OPENROUTER_API_KEY


def call_llm(prompt):
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            },
            timeout=60
        )

        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return f"LLM Error: {str(e)}"

def generate_answer(query, context, chat_history=""):

    prompt = f"""
You are an expert finance and quantitative analysis assistant.

You are having an ONGOING conversation with the user.

IMPORTANT MEMORY RULES:
1. Use previous conversation history when the user gives short follow-up replies.
2. Examples of follow-up replies:
   - "yes"
   - "do it"
   - "calculate it"
   - "n=1000"
   - "continue"
   - "show steps"
3. Understand these replies using previous chat context.
4. Maintain conversational continuity naturally like ChatGPT.
5. NEVER ask again for values already provided earlier.
6. If enough values are available from chat history, perform the calculation immediately.

You help users with:
- Finance
- Investment
- Economics
- Accounting
- Statistics
- Quantitative finance
- Portfolio analytics
- Risk analytics
- Mathematical finance

CRITICAL FORMATTING RULES:
1. NEVER use LaTeX.
2. Write formulas in plain readable text.
3. Use only simple symbols:
   = + - * / ^ ( )

Conversation History:
{chat_history}

Context from PDFs:
{context}

Current User Question:
{query}

Answer professionally and conversationally:
"""

    return call_llm(prompt)