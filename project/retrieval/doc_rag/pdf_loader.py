"""PDF loader powered by Azure Document Intelligence.

Replaces the dbgpt PDFKnowledge parser with Azure Document Intelligence layout
analysis so we can extract high-fidelity text plus structured metadata.
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, ParagraphRole

from dbgpt.core import Chunk, Document

from . import logger


class MetadataAwarePDFLoader:
    """Load PDFs via Azure Document Intelligence and preserve metadata.
    
    Features:
    - Uses Azure Document Intelligence `prebuilt-layout` (configurable) to read PDFs
    - Emits paragraph-level metadata including bounding boxes and headings
    - Injects inferred document metadata (title, page count, etc.) for later retrieval
    """
    
    def __init__(
        self,
        azure_endpoint: str,
        azure_api_key: str,
        azure_model_id: str = "prebuilt-layout",
        language: str = "en"
    ):
        """Initialize the PDF loader.
        
        Args:
            azure_endpoint: Azure Document Intelligence endpoint URL
            azure_api_key: API key for the Document Intelligence resource
            azure_model_id: Model identifier to run (default: prebuilt-layout)
            language: Locale hint passed to the analyzer
        """
        if not azure_endpoint or not azure_api_key:
            raise ValueError("Azure endpoint and API key must be provided for PDF loading.")
        
        self.azure_endpoint = azure_endpoint
        self.azure_api_key = azure_api_key
        self.azure_model_id = azure_model_id
        self.language = language
        self._client = DocumentIntelligenceClient(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(azure_api_key)
        )
    
    def load(
        self,
        file_path: str,
        doc_metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> Tuple[List[Chunk], Dict[str, Any]]:
        """Load PDF and create chunks enriched with metadata.
        
        Args:
            file_path: Path to the PDF file
            doc_metadata: Optional overrides for document-level metadata
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            Tuple of (chunks, document metadata)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        logger.info(
            "Loading PDF via Azure Document Intelligence",
            extra={"props": {"file": str(file_path), "model": self.azure_model_id}},
        )
        
        analyze_result = self._analyze_document(file_path)
        base_metadata = self._build_doc_metadata(file_path, analyze_result, doc_metadata)
        documents = self._build_documents(analyze_result)
        
        logger.info(
            "Extracted paragraphs from PDF",
            extra={
                "props": {
                    "paragraphs": len(documents),
                    "page_count": base_metadata.get("page_count", 0),
                }
            },
        )
        
        chunks = self._documents_to_chunks(documents, base_metadata, chunk_size, chunk_overlap)
        
        logger.info(
            "Created chunks with metadata",
            extra={"props": {"chunk_count": len(chunks)}},
        )
        
        return chunks, base_metadata
    
    def _analyze_document(self, file_path: Path) -> AnalyzeResult:
        """Call Azure Document Intelligence to analyze the PDF."""
        with open(file_path, "rb") as pdf_file:
            poller = self._client.begin_analyze_document(
                model_id=self.azure_model_id,
                body=pdf_file,
                content_type="application/pdf",
                locale=self.language
            )
            return poller.result()
    
    def _build_doc_metadata(
        self,
        file_path: Path,
        analyze_result: AnalyzeResult,
        overrides: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Construct document-level metadata from the analysis result."""
        metadata: Dict[str, Any] = {
            "source": str(file_path),
            "file_name": file_path.name,
            "page_count": len(analyze_result.pages or []),
            "model_id": self.azure_model_id,
            "api_version": analyze_result.api_version,
            "language": self.language,
        }
        
        languages = []
        if analyze_result.languages:
            for language in analyze_result.languages:
                languages.append({
                    "locale": language.locale,
                    "confidence": language.confidence,
                })
            metadata["detected_languages"] = languages
            # Prefer detected language if none was specified explicitly
            if languages and metadata.get("language") in (None, "", "en"):
                metadata["language"] = languages[0]["locale"]
        
        metadata["title"] = self._infer_title(analyze_result, file_path)
        metadata["sections"] = self._collect_section_headings(analyze_result)
        
        if overrides:
            metadata.update(overrides)
        
        return metadata
    
    def _build_documents(self, analyze_result: AnalyzeResult) -> List[Document]:
        """Convert Azure paragraphs into dbgpt Document objects."""
        documents: List[Document] = []
        
        paragraphs = analyze_result.paragraphs or []
        if not paragraphs and analyze_result.pages:
            # Fallback: aggregate page lines when paragraph data is missing
            for page in analyze_result.pages:
                content = "\n".join(line.content for line in page.lines or [])
                metadata = {
                    "page": page.page_number,
                    "type": "text",
                    "paragraph_role": "page",
                }
                documents.append(Document(content=content, metadata=metadata))
            return documents
        
        for paragraph in paragraphs:
            content = (paragraph.content or "").strip()
            if not content:
                continue
            
            page_number = 1
            polygon = None
            if paragraph.bounding_regions:
                region = paragraph.bounding_regions[0]
                page_number = region.page_number
                polygon = self._serialize_polygon(region.polygon)
            
            metadata = {
                "page": page_number,
                "type": "text",
                "paragraph_role": (
                    paragraph.role.value
                    if isinstance(paragraph.role, ParagraphRole)
                    else paragraph.role
                ) or "text",
            }
            if polygon:
                metadata["bounding_polygon"] = polygon
            if paragraph.spans:
                metadata["span_offset"] = paragraph.spans[0].offset
                metadata["span_length"] = paragraph.spans[0].length
            if paragraph.role == ParagraphRole.SECTION_HEADING:
                metadata["section_heading"] = content
            
            documents.append(Document(content=content, metadata=metadata))
        
        return documents
    
    def _collect_section_headings(self, analyze_result: AnalyzeResult) -> List[str]:
        """Collect section headings detected by Azure."""
        sections: List[str] = []
        for paragraph in analyze_result.paragraphs or []:
            if paragraph.role == ParagraphRole.SECTION_HEADING:
                heading = (paragraph.content or "").strip()
                if heading:
                    sections.append(heading)
        return sections
    
    def _infer_title(self, analyze_result: AnalyzeResult, file_path: Path) -> str:
        """Attempt to infer a reasonable title from the analysis output."""
        for paragraph in analyze_result.paragraphs or []:
            if paragraph.role in (ParagraphRole.SECTION_HEADING, ParagraphRole.PAGE_HEADER):
                heading = (paragraph.content or "").strip()
                if heading:
                    return heading
        return file_path.stem
    
    def _serialize_polygon(self, polygon: Optional[List[float]]) -> Optional[List[float]]:
        """Convert polygon coordinates into a serializable list."""
        if not polygon:
            return None
        return [float(point) for point in polygon]
    
    def _documents_to_chunks(
        self,
        documents: List[Document],
        base_metadata: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Chunk]:
        """Convert documents to chunks with enhanced metadata."""
        chunks = []
        current_section = base_metadata.get("title", "Introduction")
        
        for doc in documents:
            content = doc.content
            if not content or not content.strip():
                continue
            
            doc_meta = doc.metadata if hasattr(doc, "metadata") and doc.metadata else {}
            page = doc_meta.get("page", 1)
            content_type = doc_meta.get("type", "text")
            
            explicit_section = doc_meta.get("section_heading") or doc_meta.get("section")
            if explicit_section:
                current_section = explicit_section
            else:
                detected_section = self._detect_section(content)
                if detected_section:
                    current_section = detected_section
            
            content_chunks = self._split_content(content, chunk_size, chunk_overlap)
            
            for i, chunk_content in enumerate(content_chunks):
                chunk_metadata = {
                    **base_metadata,
                    "page": page,
                    "type": content_type,
                    "section": current_section,
                    "chunk_index": i,
                    "total_chunks_in_doc": len(content_chunks),
                    "paragraph_role": doc_meta.get("paragraph_role", "text"),
                }
                if doc_meta.get("bounding_polygon"):
                    chunk_metadata["bounding_polygon"] = doc_meta["bounding_polygon"]
                if doc_meta.get("span_offset") is not None:
                    chunk_metadata["span_offset"] = doc_meta["span_offset"]
                    chunk_metadata["span_length"] = doc_meta.get("span_length")
                
                chunks.append(
                    Chunk(
                        content=chunk_content,
                        metadata=chunk_metadata
                    )
                )
        
        return chunks
    
    def _detect_section(self, content: str) -> Optional[str]:
        """Detect section headers from content."""
        patterns = [
            r"^(\d+(?:\.\d+)?)\s+([A-Z][A-Za-z\s]+)",
            r"^([A-Z]{2,}(?:\s+[A-Z]{2,})*)\s*$",
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\n",
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
        """Split content into overlapping chunks."""
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            
            if end < len(content):
                search_start = max(start, end - int(chunk_size * 0.2))
                sentence_ends = [
                    content.rfind(". ", search_start, end),
                    content.rfind("? ", search_start, end),
                    content.rfind("! ", search_start, end),
                    content.rfind("\n", search_start, end),
                ]
                best_end = max(sentence_ends)
                if best_end > search_start:
                    end = best_end + 1
            
            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            start = end - chunk_overlap if end < len(content) else len(content)
        
        return chunks
