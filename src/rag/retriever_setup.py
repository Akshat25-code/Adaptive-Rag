"""
Retriever setup and vector store configuration.
"""

import logging
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

embeddings = OpenAIEmbeddings()

FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_index")

# Global state
_faiss_vectorstore = None
_document_count = 0


def _save_to_disk():
    """Persist FAISS index to disk."""
    global _faiss_vectorstore
    if _faiss_vectorstore is not None:
        _faiss_vectorstore.save_local(FAISS_INDEX_DIR)
        logger.info("FAISS index saved to %s", FAISS_INDEX_DIR)


def _count_docs():
    """Count documents in FAISS store safely across versions."""
    global _faiss_vectorstore
    try:
        return len(_faiss_vectorstore.docstore._dict)
    except AttributeError:
        try:
            return _faiss_vectorstore.index.ntotal
        except Exception:
            return 0


def _load_from_disk():
    """Load FAISS index from disk if exists."""
    global _faiss_vectorstore, _document_count
    if os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss")):
        try:
            _faiss_vectorstore = FAISS.load_local(
                FAISS_INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            _document_count = _count_docs()
            logger.info("Loaded FAISS index from disk (%d chunks)", _document_count)
            return True
        except Exception as e:
            logger.error("Failed to load FAISS index: %s", e)
    return False


# Load on startup
_load_from_disk()


def retriever_chain(chunks: list[Document]):
    """
    Add document chunks to FAISS vector store.
    Accumulates across uploads. Persists to disk.
    """
    global _faiss_vectorstore, _document_count

    try:
        new_store = FAISS.from_documents(documents=chunks, embedding=embeddings)

        if _faiss_vectorstore is not None:
            _faiss_vectorstore.merge_from(new_store)
            logger.info("Merged %d chunks into existing store", len(chunks))
        else:
            _faiss_vectorstore = new_store
            logger.info("Created new FAISS store with %d chunks", len(chunks))

        _document_count += len(chunks)
        _save_to_disk()
        return True
    except Exception as e:
        logger.error("Error storing documents in FAISS: %s", e)
        return False


def get_retriever():
    """Get retriever tool connected to FAISS vector store."""
    global _faiss_vectorstore

    try:
        if _faiss_vectorstore is not None:
            retriever = _faiss_vectorstore.as_retriever()
            logger.info("Using FAISS vectorstore (%d total chunks)", _document_count)
        else:
            logger.info("No documents uploaded, creating dummy vectorstore")
            dummy_doc = Document(
                page_content="No documents have been uploaded yet. Please upload a document first.",
                metadata={"source": "initialization"},
            )
            _faiss_vectorstore = FAISS.from_documents(documents=[dummy_doc], embedding=embeddings)
            retriever = _faiss_vectorstore.as_retriever()

        if os.path.exists("description.txt"):
            with open("description.txt", encoding="utf-8") as f:
                description = f.read()
        else:
            description = None

        retriever_tool = create_retriever_tool(
            retriever,
            "retriever_customer_uploaded_documents",
            f"Use this tool **only** to answer questions about: {description}\n"
            "Don't use this tool to answer anything else.",
        )

        return retriever_tool

    except Exception as e:
        logger.error("Error initializing retriever: %s", e)
        raise


def get_document_count() -> int:
    """Return total number of document chunks stored."""
    return _document_count


def clear_documents():
    """Clear all documents from vector store and disk."""
    global _faiss_vectorstore, _document_count
    _faiss_vectorstore = None
    _document_count = 0
    # Remove index files
    if os.path.exists(FAISS_INDEX_DIR):
        import shutil

        shutil.rmtree(FAISS_INDEX_DIR)
    logger.info("Cleared all documents from FAISS store and disk")
