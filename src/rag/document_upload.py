"""
Document upload and processing module.
"""

import logging
import os
import tempfile

from fastapi import File, HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.retriever_setup import retriever_chain
from src.tools.common_tools import enhance_description_with_llm

logger = logging.getLogger(__name__)


def documents(description: str, file: UploadFile = File(...)):
    """
    Process and upload a document for RAG.

    Args:
        description: User-provided document description.
        file: The uploaded file (PDF or TXT).

    Returns:
        Boolean indicating success of the upload process.
    """
    filename = file.filename
    logger.info("Processing upload: %s", filename)

    if not filename.endswith(".pdf") and not filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    file_bytes = file.file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    if filename.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path, encoding="utf-8")

    try:
        docs = loader.load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading file: {e}")
    finally:
        os.unlink(tmp_path)

    # Enhance description using LLM
    description_llm = enhance_description_with_llm(description)

    with open("description.txt", "w", encoding="utf-8") as f:
        f.write(description_llm)

    logger.info("Enhanced description saved for: %s", filename)

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks", len(chunks))

    return retriever_chain(chunks)
