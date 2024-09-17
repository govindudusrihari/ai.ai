from fastapi import FastAPI, Request
from haystack.pipelines import RAG
from haystack.utils import load_from_checkpoint

app = FastAPI()

# Load your pre-trained LLM model
model = load_from_checkpoint("path/to/your/model.ckpt")

# Create a RAG pipeline
rag_pipeline = RAG( retriever=your_retriever, generator=model)

@app.post("/ask")
async def ask(request: Request):
    query = await request.json()
    results = rag_pipeline.run(query)
    return results

if __name__ == "__main__":
    app.run(debug=True)
