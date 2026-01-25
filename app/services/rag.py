from typing import List, Dict, Any
import os

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from app.config import get_settings
from app.services.embeddings import get_embeddings
from app.services.llm import get_llm


def _format_docs(docs: List[Document]) -> str:
    """Format documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


class RAGService:
    """
    RAG (Retrieval-Augmented Generation) service.
    Handles document indexing and question-answering using LCEL.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._vectorstore = None
        self._rag_chain = None
        self._retriever = None
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # RAG prompt template using ChatPromptTemplate (LCEL-compatible)
        # Loaded from environment variable or default
        self.prompt = ChatPromptTemplate.from_template(self.settings.rag_prompt_template)
    
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
    def retriever(self):
        """Lazy-load retriever."""
        if self._retriever is None:
            self._retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
        return self._retriever
    
    @property
    def rag_chain(self):
        """Lazy-load RAG chain using LCEL."""
        if self._rag_chain is None:
            # Build LCEL chain: retrieve -> format -> prompt -> llm -> parse
            self._rag_chain = (
                RunnablePassthrough.assign(
                    context=lambda x: _format_docs(self.retriever.invoke(x["question"]))
                )
                | self.prompt
                | get_llm()
                | StrOutputParser()
            )
        return self._rag_chain
    
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
        Search for answer using RAG with LCEL.
        
        Args:
            question: User question
            
        Returns:
            Dict with 'answer' and 'sources'
        """
        # Get source documents separately for transparency
        source_docs = self.retriever.invoke(question)
        
        # Run the RAG chain
        answer = self.rag_chain.invoke({"question": question})
        
        # Extract source information
        sources = []
        for doc in source_docs:
            source_info = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)
        
        return {
            "answer": answer,
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
