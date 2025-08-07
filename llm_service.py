from openai import OpenAI
import json
from config import Config

client = OpenAI(
    api_key=Config.OPENROUTER_API_KEY,
    base_url=Config.OPENROUTER_BASE,
)

PROMPT_TEMPLATE = """
Analyze the insurance policy clauses to answer this query:
{query}

Relevant context:
{context}

Respond in JSON format:
{{
  "answer": "...",
  "conditions": [...],
  "explanation": "...",
  "clause_ids": [...]
}}
"""

def generate_response(query: str, context: str) -> str:
    try:
        response = client.chat.completions.create(
            model=Config.OPENROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": PROMPT_TEMPLATE.format(query=query, context=context)
                }
            ],
            response_format="json",
            max_tokens=1024,
        )
        raw_output = response.choices[0].message.content
        print("\n🔍 Raw LLM Response:", raw_output)  # Log it for debugging

        if not raw_output:
            raise ValueError("LLM returned empty response")

        return json.loads(raw_output)["answer"]

    except Exception as e:
        print("❌ Error in generate_response:", str(e))
        return "Failed to generate response due to an error."
