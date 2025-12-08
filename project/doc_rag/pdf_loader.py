"""PDF loader with page-level metadata extraction.

Wraps dbgpt PDFKnowledge with enhanced metadata for page navigation.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from dbgpt.core import Chunk, Document
from dbgpt_ext.rag.knowledge.pdf import PDFKnowledge

from . import logger


class MetadataAwarePDFLoader:
    """PDF loader that extracts and preserves page-level metadata.
    
    Features:
    - Uses dbgpt PDFKnowledge for PDF parsing
    - Adds page number, content type (text/table), and section detection
    - Injects document-level metadata (title, authors, etc.)
    - Preserves metadata through chunking for later retrieval
    """
    
    def __init__(self, language: str = "en"):
        """Initialize the PDF loader.
        
        Args:
            language: Language for PDF processing ("en" or "zh")
        """
        self.language = language
    
    def load(
        self, 
        file_path: str, 
        doc_metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> List[Chunk]:
        """Load PDF and create chunks with page-level metadata.
        
        Args:
            file_path: Path to the PDF file
            doc_metadata: Optional document-level metadata (title, authors, etc.)
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List of Chunk objects with rich metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        logger.info(f"Loading PDF", extra={"props": {"file": str(file_path)}})
        
        # Load base metadata
        base_metadata = doc_metadata or {}
        base_metadata["source"] = str(file_path)
        base_metadata["file_name"] = file_path.name
        
        # Use dbgpt PDFKnowledge to extract documents
        pdf_knowledge = PDFKnowledge(
            file_path=str(file_path),
            language=self.language
        )
        
        # Load documents from PDF
        documents = pdf_knowledge._load()
        
        logger.info(f"Extracted documents from PDF", extra={"props": {"count": len(documents)}})
        
        # Convert documents to chunks with enhanced metadata
        chunks = self._documents_to_chunks(documents, base_metadata, chunk_size, chunk_overlap)
        
        logger.info(f"Created chunks with metadata", extra={"props": {"count": len(chunks)}})
        
        return chunks
    
    def _documents_to_chunks(
        self,
        documents: List[Document],
        base_metadata: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Chunk]:
        """Convert documents to chunks with enhanced metadata.
        
        Args:
            documents: List of Document objects from PDFKnowledge
            base_metadata: Document-level metadata to inject
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        current_section = "Introduction"  # Default section
        
        for doc in documents:
            content = doc.content
            if not content or not content.strip():
                continue
            
            # Extract metadata from document
            doc_meta = doc.metadata if hasattr(doc, 'metadata') and doc.metadata else {}
            
            page = doc_meta.get("page", 1)
            content_type = doc_meta.get("type", "text")
            
            # Detect section from content (simple heuristic)
            detected_section = self._detect_section(content)
            if detected_section:
                current_section = detected_section
            
            # Split content into chunks if too long
            content_chunks = self._split_content(content, chunk_size, chunk_overlap)
            
            for i, chunk_content in enumerate(content_chunks):
                # Create rich metadata
                chunk_metadata = {
                    **base_metadata,
                    "page": page,
                    "type": content_type,
                    "section": current_section,
                    "chunk_index": i,
                    "total_chunks_in_doc": len(content_chunks),
                }
                
                chunk = Chunk(
                    content=chunk_content,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)
        
        return chunks
    
    def _detect_section(self, content: str) -> Optional[str]:
        """Detect section headers from content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Detected section name or None
        """
        # Common section patterns
        patterns = [
            # Numbered sections: 1. Introduction, 2.1 Methods
            r'^(\d+(?:\.\d+)?)\s+([A-Z][A-Za-z\s]+)',
            # ALL CAPS headers
            r'^([A-Z]{2,}(?:\s+[A-Z]{2,})*)\s*$',
            # Title case headers followed by newline
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\n',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content[:200], re.MULTILINE)
            if match:
                section = match.group(2) if len(match.groups()) > 1 else match.group(1)
                return section.strip()
        
        return None
    
    def _split_content(
        self, 
        content: str, 
        chunk_size: int, 
        chunk_overlap: int
    ) -> List[str]:
        """Split content into chunks with overlap.
        
        Args:
            content: Text content to split
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of content chunks
        """
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence end within last 20% of chunk
                search_start = max(start, end - int(chunk_size * 0.2))
                sentence_ends = [
                    content.rfind('. ', search_start, end),
                    content.rfind('? ', search_start, end),
                    content.rfind('! ', search_start, end),
                    content.rfind('\n', search_start, end),
                ]
                best_end = max(sentence_ends)
                if best_end > search_start:
                    end = best_end + 1
            
            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            # Move start with overlap
            start = end - chunk_overlap if end < len(content) else len(content)
        
        return chunks
    
    @staticmethod
    def load_metadata_from_json(json_path: str) -> Dict[str, Any]:
        """Load document metadata from a JSON file.
        
        Args:
            json_path: Path to the metadata JSON file
            
        Returns:
            Dictionary of metadata
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
