import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

EMBEDDING_MODEL_NAME = "dangvantuan/sentence-camembert-base"

FAISS_INDEX_PATH = "data/faiss_index"
DATASET_NAME = "harvard-lil/cold-french-law"

LLM_MODEL = "ministral-3:3b-instruct-2512-q4_K_M"
LLM_TEMPERATURE = 0.0

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K_RETRIEVAL = 5
