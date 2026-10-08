"""
Document Indexer module for parsing PDF documents, generating semantic chunks,
computing embeddings via Gemini, and caching vectors for fast retrieval.
"""

import os
import json
import hashlib
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import numpy as np
import pypdf
from google import genai


@dataclass
class DocumentChunk:
    chunk_id: str
    page_number: int
    section: str
    text: str
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(**data)


class DocumentIndexer:
    """
    Handles PDF loading, text extraction, intelligent chunking,
    Gemini vector embeddings, and persistent local caching.
    """

    KNOWN_SECTION_HEADERS = [
        "BREAKFAST & TIFFIN",
        "RICE & MEALS",
        "BIRYANI & FRIED SPECIALS",
        "VEG & NON-VEG SPECIALS",
        "BREADS & DRINKS",
        "MENU CARD",
        "NILA HOTEL",
    ]

    def __init__(
        self,
        pdf_path: str = "nila.pdf",
        client: Optional[genai.Client] = None,
        embedding_model: str = "gemini-embedding-001",
        cache_dir: str = ".cache",
    ):
        self.pdf_path = pdf_path
        self.client = client
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        self.pages_text: List[str] = []
        self.chunks: List[DocumentChunk] = []
        self.sections_found: List[str] = []
        self.file_hash: str = ""

    def load_and_index(self, force_refresh: bool = False) -> List[DocumentChunk]:
        """
        Main entry point to parse PDF, chunk text, and generate or load cached embeddings.
        """
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(
                f"Document not found at '{self.pdf_path}'. "
                f"Please ensure '{os.path.basename(self.pdf_path)}' is placed in the project root."
            )

        self.file_hash = self._compute_file_hash(self.pdf_path)
        cache_path = os.path.join(self.cache_dir, f"index_{self.file_hash[:12]}.json")

        # 1. Parse PDF text
        self._extract_pdf_pages()

        # 2. Check cache first
        if not force_refresh and os.path.exists(cache_path):
            try:
                self._load_cache(cache_path)
                return self.chunks
            except Exception as e:
                print(f"[Indexer Warning] Cache corrupted or outdated ({e}). Rebuilding index...")

        # 3. Create structured chunks from pages
        self._create_chunks()

        # 4. Generate embeddings
        self._generate_embeddings()

        # 5. Save to disk cache
        self._save_cache(cache_path)

        return self.chunks

    def _compute_file_hash(self, path: str) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _extract_pdf_pages(self) -> None:
        """Extract text from each page using pypdf."""
        reader = pypdf.PdfReader(self.pdf_path)
        self.pages_text = []
        for idx, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            # Clean up whitespace while preserving newlines
            cleaned = "\n".join(
                line.strip() for line in raw_text.splitlines() if line.strip()
            )
            self.pages_text.append(cleaned)

    def _detect_section(self, line: str) -> Optional[str]:
        """Detect if a line is a section heading."""
        cleaned = line.strip().upper()
        for header in self.KNOWN_SECTION_HEADERS:
            if header in cleaned:
                return header
        # Generic heuristic: all caps line with 4+ chars and no numbers/currency
        if cleaned.isupper() and len(cleaned) >= 4 and not re.search(r"[\d₹]", cleaned):
            return cleaned
        return None

    def _create_chunks(self) -> None:
        """
        Create multi-granularity chunks:
        1. Section-level chunks grouping items under their respective category.
        2. Page-level summary chunks.
        3. Fine-grained item lines for exact lookup.
        """
        self.chunks = []
        self.sections_found = []
        chunk_count = 0

        for page_idx, page_text in enumerate(self.pages_text):
            page_num = page_idx + 1
            lines = page_text.splitlines()
            current_section = "GENERAL"
            current_section_lines = []

            for line in lines:
                detected = self._detect_section(line)
                if detected:
                    # Flush previous section chunk if it has content
                    if current_section_lines:
                        chunk_text = f"[{current_section}]\n" + "\n".join(current_section_lines)
                        self.chunks.append(
                            DocumentChunk(
                                chunk_id=f"chk_p{page_num}_{chunk_count}",
                                page_number=page_num,
                                section=current_section,
                                text=chunk_text.strip(),
                            )
                        )
                        chunk_count += 1
                        current_section_lines = []

                    current_section = detected
                    if current_section not in self.sections_found:
                        self.sections_found.append(current_section)
                else:
                    current_section_lines.append(line)

            # Flush remaining lines in the page
            if current_section_lines:
                chunk_text = f"[{current_section}]\n" + "\n".join(current_section_lines)
                self.chunks.append(
                    DocumentChunk(
                        chunk_id=f"chk_p{page_num}_{chunk_count}",
                        page_number=page_num,
                        section=current_section,
                        text=chunk_text.strip(),
                    )
                )
                chunk_count += 1

            # Page overview chunk
            self.chunks.append(
                DocumentChunk(
                    chunk_id=f"chk_page_full_{page_num}",
                    page_number=page_num,
                    section=f"PAGE_{page_num}_FULL",
                    text=f"=== COMPLETE PAGE {page_num} ===\n{page_text}",
                )
            )

    def _generate_embeddings(self) -> None:
        """
        Generate embeddings for all chunks using Gemini API.
        If API is unavailable, gracefully mark as None to allow fallback retrieval.
        """
        if not self.client:
            print("[Indexer Info] No Gemini client provided. Semantic embeddings skipped.")
            return

        texts = [chunk.text for chunk in self.chunks]
        print(f"[Indexer Info] Generating embeddings for {len(texts)} chunks using {self.embedding_model}...")

        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                for j, text in enumerate(batch):
                    idx = i + j
                    res = self.client.models.embed_content(
                        model=self.embedding_model,
                        contents=text,
                    )
                    if res.embeddings and len(res.embeddings) > 0:
                        self.chunks[idx].embedding = list(res.embeddings[0].values)
            except Exception as e:
                print(f"[Indexer Warning] Embedding generation failed for batch {i}: {e}")
                # We do not crash; hybrid retriever will fall back to keyword/TF-IDF scoring
                break

    def _save_cache(self, cache_path: str) -> None:
        """Persist processed chunks and embeddings to disk cache."""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_data = {
            "pdf_path": self.pdf_path,
            "file_hash": self.file_hash,
            "sections_found": self.sections_found,
            "chunks": [c.to_dict() for c in self.chunks],
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

    def _load_cache(self, cache_path: str) -> None:
        """Load index chunks from cache file."""
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.sections_found = data.get("sections_found", [])
        self.chunks = [DocumentChunk.from_dict(item) for item in data.get("chunks", [])]
