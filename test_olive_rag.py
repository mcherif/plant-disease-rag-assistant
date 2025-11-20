import os
from unittest.mock import MagicMock
from src.llm.rag_pipeline import RAGPipeline, RetrievalConfig

# Mock the _generate method to avoid OpenAI API key dependency in CI
def mock_generate(self, prompt, model=None, **kwargs):
    return "This is a mocked answer for testing purposes."

# Apply mock if OPENAI_API_KEY is not set
if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY not set. Mocking LLM generation.")
    RAGPipeline._generate = mock_generate

cfg = RetrievalConfig(index_dir='models/index/kb-faiss-bge', top_k=3, device='cpu')
rag = RAGPipeline(cfg)

# Test retrieval for Peacock spot
result = rag.answer('What is Peacock spot on olive and how can I treat it?', plant='Olive', disease='Peacock spot')
print('Query: What is Peacock spot on olive and how can I treat it?')
print(f'\nAnswer: {result.get("answer", "No answer")}')
print(f'\nRetrieved {len(result.get("retrieved", []))} chunks:')
for i, doc in enumerate(result.get('retrieved', []), 1):
    meta = doc.get('meta', {})
    title = meta.get('title', 'N/A')
    plant = meta.get('plant', 'N/A')
    disease = meta.get('disease', 'N/A')
    text = meta.get('text', '')[:100]
    print(f'  {i}. {title} (plant={plant}, disease={disease})')
    print(f'     {text}...')
