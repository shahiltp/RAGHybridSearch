from fastapi import FastAPI
from app.api.routes_ask import router as ask_router

app = FastAPI(title="LegalMind Hybrid RAG")

app.include_router(ask_router)
