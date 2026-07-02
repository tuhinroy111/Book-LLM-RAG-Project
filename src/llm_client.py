import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

load_dotenv()


class LLMClient:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)

    def generate_with_retry(
        self,
        prompt: str,
        model_name: str = "gemini-2.5-flash",
        retries: int = 3,
        delay: int = 5,
    ):
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return response.text

            except APIError as e:
                if e.code == 429 and attempt < retries - 1:
                    print(f"Rate limit hit. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise

    def generate_response(self, query: str, context: str):
        prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.
If the answer is not present in the context, say:
"I couldn't find that information in the document."

Context:
{context}

Question:
{query}

Answer:
"""

        return self.generate_with_retry(prompt)