import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time
from google.genai import errors

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

emb = client.models.embed_content(
    model=os.environ["EMBEDDING_MODEL"],
    contents="Bonjour, hello",
    config=types.EmbedContentConfig(output_dimensionality=768),
)
print("embedding dimensions:", len(emb.embeddings[0].values))

reply = client.models.generate_content(
    model=os.environ["LLM_MODEL"],
    contents="Say hi in five words.",
)
print("chat reply:", reply.text)

def generate_with_retry(client, model, contents, attempts=5):
    for i in range(attempts):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except errors.ServerError as e:
            wait = 2 ** i
            print(f"server busy ({e.code}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("model unavailable after retries")

reply = generate_with_retry(client, os.environ["LLM_MODEL"], "Say hi in five words.")
print("chat reply:", reply.text)
