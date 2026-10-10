import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

emb = client.models.embed_content(
    model=os.environ["EMBEDDING_MODEL"],
    contents="Bonjour, hello",
)
print("embedding dimensions:", len(emb.embeddings[0].values))

reply = client.models.generate_content(
    model=os.environ["LLM_MODEL"],
    contents="Say hi in five words.",
)
print("chat reply:", reply.text)
