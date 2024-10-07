import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Optional, Dict, Callable
import os
import pickle
from datetime import datetime
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class PyMuPDFReader(BaseReader):
    def __init__(
            self,
            input_dir: Optional[str] = None,
            input_files: Optional[List[Path]] = None,
            exclude_hidden: bool = True,
            file_metadata: Optional[Callable[[str], Dict]] = None,
            **kwargs
    ):
        """
        Initialize the PyMuPDFReader.

        Args:
            input_dir: Directory to read files from.
            input_files: Specific files to read.
            exclude_hidden: Whether to exclude hidden files.
            file_metadata: A function to extract file metadata.
            kwargs: Additional arguments for the BaseReader.
        """
        super().__init__(**kwargs)  # Initialize the base class
        self.exclude_hidden = exclude_hidden
        self.input_dir = input_dir
        self.input_files = input_files or self._collect_files(input_dir)
        self.file_metadata = file_metadata

    def _collect_files(self, input_dir: Optional[str]) -> List[Path]:
        """Collect all PDF files from the directory."""
        if input_dir is None:
            return []

        pdf_files = []
        for path in Path(input_dir).rglob("*.pdf"):
            if self.exclude_hidden and path.name.startswith('.'):
                continue
            pdf_files.append(path)
        return pdf_files

    def _extract_metadata(self, file_path: Path) -> Dict:
        """Extract metadata for a given PDF file."""
        file_stat = file_path.stat()
        return {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_type': 'application/pdf',
            'file_size': file_stat.st_size,
            'creation_date': datetime.fromtimestamp(file_stat.st_ctime).strftime('%Y-%m-%d'),
            'last_modified_date': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d'),
            'last_accessed_date': datetime.fromtimestamp(file_stat.st_atime).strftime('%Y-%m-%d'),
        }

    def load_data(self, show_progress: bool = False, raise_on_error: bool = False, **kwargs) -> List[Document]:
        """Load data from PDF files in the directory using PyMuPDF."""
        documents = []
        total_files = len(self.input_files)

        for index, input_file in enumerate(self.input_files):
            try:
                if show_progress:
                    print(f"Processing file {index + 1}/{total_files}: {input_file}")

                with fitz.open(input_file) as doc:
                    text = ""
                    for page_num in range(doc.page_count):
                        page = doc.load_page(page_num)
                        text += page.get_text()

                    # Extract metadata
                    metadata = self._extract_metadata(input_file)
                    if self.file_metadata:
                        metadata.update(self.file_metadata(str(input_file)))

                    # Define additional fields
                    excluded_embed_metadata_keys = [
                        'file_name', 'file_type', 'file_size', 'creation_date',
                        'last_modified_date', 'last_accessed_date'
                    ]
                    excluded_llm_metadata_keys = excluded_embed_metadata_keys
                    relationships = {}
                    mimetype = 'text/plain'
                    start_char_idx = None
                    end_char_idx = None
                    text_template = '{metadata_str}\n\n{content}'
                    metadata_template = '{key}: {value}'
                    metadata_separator = '\n'

                    # Create a Document object with text, metadata, and additional fields
                    document = Document(
                        text=text,
                        metadata=metadata,
                        excluded_embed_metadata_keys=excluded_embed_metadata_keys,
                        excluded_llm_metadata_keys=excluded_llm_metadata_keys,
                        relationships=relationships,
                        mimetype=mimetype,
                        start_char_idx=start_char_idx,
                        end_char_idx=end_char_idx,
                        text_template=text_template,
                        metadata_template=metadata_template,
                        metadata_separator=metadata_separator
                    )
                    documents.append(document)

            except Exception as e:
                print(f"Failed to load PDF file {input_file} with error: {e}. Skipping...")
                if raise_on_error:
                    raise e

        return documents