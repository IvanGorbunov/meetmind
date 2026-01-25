from typing import List, Dict, Any
import os

from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from app.config import get_settings
from app.services.embeddings import get_embeddings
from app.services.llm import get_llm


class RAGService:
    """
    RAG (Retrieval-Augmented Generation) service.
    Handles document indexing and question-answering.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._vectorstore = None
        self._qa_chain = None
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # RAG prompt template
        self.prompt_template = PromptTemplate(
            template="""Ты ассистент для анализа рабочих созвонов.
Отвечай только на основе предоставленного контекста.
Если информации нет в контексте, скажи об этом честно.

Контекст:
{context}

Вопрос: {question}

Ответ:""",
            input_variables=["context", "question"]
        )
    
    @property
    def vectorstore(self) -> Chroma:
        """Lazy-load vector store."""
        if self._vectorstore is None:
            # Ensure directory exists
            os.makedirs(self.settings.chroma_persist_directory, exist_ok=True)
            
            self._vectorstore = Chroma(
                persist_directory=self.settings.chroma_persist_directory,
                embedding_function=get_embeddings()
            )
        return self._vectorstore
    
    @property
    def qa_chain(self) -> RetrievalQA:
        """Lazy-load QA chain."""
        if self._qa_chain is None:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            self._qa_chain = RetrievalQA.from_chain_type(
                llm=get_llm(),
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": self.prompt_template},
                return_source_documents=True
            )
        return self._qa_chain
    
    def index_document(self, content: str, metadata: Dict[str, Any] = None) -> int:
        """
        Index a document into the vector store.
        
        Args:
            content: Document text content
            metadata: Optional metadata dict (filename, date, etc.)
            
        Returns:
            Number of chunks indexed
        """
        # Split into chunks
        chunks = self.text_splitter.split_text(content)
        
        # Create documents with metadata
        documents = [
            Document(page_content=chunk, metadata=metadata or {})
            for chunk in chunks
        ]
        
        # Add to vector store
        self.vectorstore.add_documents(documents)
        
        return len(chunks)
    
    def search(self, question: str) -> Dict[str, Any]:
        """
        Search for answer using RAG.
        
        Args:
            question: User question
            
        Returns:
            Dict with 'answer' and 'sources'
        """
        result = self.qa_chain.invoke({"query": question})
        
        # Extract source information
        sources = []
        for doc in result.get("source_documents", []):
            source_info = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)
        
        return {
            "answer": result["result"],
            "sources": sources
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        collection = self.vectorstore._collection
        return {
            "total_documents": collection.count()
        }


# Singleton instance
_rag_service = None


def get_rag_service() -> RAGService:
    """Get singleton RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
