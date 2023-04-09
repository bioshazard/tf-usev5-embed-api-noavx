from langchain.embeddings import TensorflowHubEmbeddings
url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
print("Loading model... (can take a minute)",)
embeddings = TensorflowHubEmbeddings(model_url=url)
print("DONE")

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class EmbedPost(BaseModel):
    texts: List[str]

@app.post("/embed/")
async def embed(embed: EmbedPost):
    embeds = embeddings.embed_documents(embed.texts)
    return { "texts": embed.texts, "embeddings": embeds }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("embed:app", host="0.0.0.0", port=8885)
