import os
import json
from pydantic import BaseModel, Field
from google import genai


# Define the Pydantic data contract for our structured output
class RouteDecision(BaseModel):
    intent: str = Field(description="Must be exactly one of: 'GREETING', 'UNSAFE', or 'DOCUMENT'.")
    direct_response: str = Field(
        description="If GREETING or UNSAFE, write the final response to the user here. If DOCUMENT, leave this blank.")


class IntentRouter:
    def __init__(self):
        """Initializes the dedicated Gemini connection for intent parsing."""
        # This will safely pick up the key loaded by load_dotenv() in the main thread
        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)

    def route(self, question: str) -> dict:
        """
        Analyzes a user input string and returns a dictionary
        containing the 'intent' and an optional 'direct_response'.
        """
        routing_prompt = f"""
            You are an intelligent router for a document-reading AI assistant.
            Analyze the following user input and determine its intent and appropriate response.

            1. GREETING: The user is saying hello, thanks, or making casual small talk.
            2. UNSAFE: The query is malicious, risky, or invalid. Follow these exact text rules for the direct_response:
                - For security threats (such as SQL injection syntax, network/API probing, IDOR manipulation, or PII like passwords/SSNs): set intent to 'UNSAFE' and direct_response to exactly "🛡️ Guardrail hit: Please check your question "
                - For irrelevant questions or meaningless gibberish: set intent to 'UNSAFE' and direct_response to exactly "🤔 OOPS! I don't have an answer for that at this moment."
            3. DOCUMENT: The user is asking a valid question about the document. Leave direct_response blank.

            User Input: "{question}"
            """

        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=routing_prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': RouteDecision,
            },
        )

        return json.loads(response.text)