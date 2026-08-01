"""
API routes for RAG operations.
"""

import json
import logging
import time

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.memory.chat_history_mongo import ChatHistory
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder
from src.rag.retriever_setup import clear_documents, get_document_count

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

_query_count = 0


@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    return {
        "status": "healthy",
        "documents_loaded": get_document_count(),
        "queries_processed": _query_count,
    }


@router.get("/rag/stats")
async def rag_stats(request: Request):
    """System statistics endpoint."""
    return {
        "document_chunks": get_document_count(),
        "total_queries": _query_count,
    }


@router.post("/rag/query")
@limiter.limit("15/minute")
async def rag_query(request: Request, req: QueryRequest):
    """
    Process a RAG query and return the result.
    """
    global _query_count

    try:
        chat_history = ChatHistory.get_session_history(req.session_id)
        await chat_history.add_message(HumanMessage(content=req.query))

        messages = await chat_history.get_messages()
        start = time.time()
        result = builder.invoke({"messages": messages})
        elapsed = round(time.time() - start, 2)

        output_text = result["messages"][-1].content
        await chat_history.add_message(AIMessage(content=output_text))

        _query_count += 1
        logger.info("Query processed in %ss: %s", elapsed, req.query[:80])

        return {
            "result": result["messages"][-1],
            "route": result.get("route", "unknown"),
            "time_seconds": elapsed,
            "source_documents": result.get("source_documents", []),
        }
    except Exception as e:
        logger.exception("Error processing query: %s", e)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.post("/rag/query/stream")
@limiter.limit("10/minute")
async def rag_query_stream(request: Request, req: QueryRequest):
    """
    Stream a RAG query response using Server-Sent Events.
    """
    global _query_count

    async def event_stream():
        try:
            chat_history = ChatHistory.get_session_history(req.session_id)
            await chat_history.add_message(HumanMessage(content=req.query))
            messages = await chat_history.get_messages()

            yield f"data: {json.dumps({'type': 'status', 'content': 'Processing query...'})}\n\n"

            result = builder.invoke({"messages": messages})
            output_text = result["messages"][-1].content
            sources = result.get("source_documents", [])

            await chat_history.add_message(AIMessage(content=output_text))

            words = output_text.split()
            chunk_size = 5
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

            done_payload = {
                "type": "done",
                "route": result.get("route", "unknown"),
                "sources": sources,
            }
            yield f"data: {json.dumps(done_payload)}\n\n"

        except Exception as e:
            logger.exception("Streaming error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    _query_count += 1
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/rag/documents/upload")
@limiter.limit("5/minute")
async def upload_file(
    request: Request, file: UploadFile = File(...), description: str = Header(..., alias="X-Description")
):
    """
    Upload a document for RAG processing.
    """
    try:
        status_upload = documents(description, file)
        return {
            "status": status_upload,
            "total_chunks": get_document_count(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.delete("/rag/documents")
async def clear_all_documents(request: Request):
    """Clear all uploaded documents from the vector store."""
    clear_documents()
    return {"status": "cleared", "document_chunks": 0}


@router.get("/rag/documents/count")
async def document_count(request: Request):
    """Get the number of document chunks in the vector store."""
    return {"document_chunks": get_document_count()}
