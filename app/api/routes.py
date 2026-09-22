from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.generation.rag_service import RAGService


router = APIRouter()

rag_service = RAGService()


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)


class Source(BaseModel):
    source: str
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]

@router.get("/health")
def health_check():
    return {
        "status": "healthy"
    }

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:
        result = rag_service.answer(request.query)

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer."
        )