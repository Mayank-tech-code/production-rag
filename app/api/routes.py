from fastapi import APIRouter
from pydantic import BaseModel

from app.generation.rag_service import RAGService


router = APIRouter()

rag_service = RAGService()


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str


@router.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    answer = rag_service.answer(request.query)

    return ChatResponse(
        answer=answer
    )