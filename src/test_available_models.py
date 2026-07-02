import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Initialize the client
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Fetch the list of accessible models
try:
    models = client.models.list()
    print("✨ Your accessible models:")
    for model in models:
        print(f"- {model.id}")
except Exception as e:
    print(f"❌ Could not retrieve models: {e}")