from typing import Dict, Union, List
import os
import logging
import pypdf
import pandas as pd
from omegaconf import DictConfig
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedUnstructuredDocument:
    """Represents a loaded document with its content and metadata."""
    content: str
    metadata: Dict[str, str]


@dataclass
class LoadedStructuredDocument:
    """Loaded structured document (CSV, Excel) with content & metadata."""
    content: pd.DataFrame
    metadata: Dict[str, str]


class LocalDocLoader:
    """Loads documents of various formats (PDF) into unified format."""
    @staticmethod
    def convert_excel_to_csv(
        excel_path: str, csv_dir: str = "./data/csv"
    ) -> List[str]:
        """Convert Excel file to CSV and save in the same directory
        
        Args:
            excel_path: Path to the Excel file
            
        Returns:
            Path to the saved CSV file
        """
        try:
            os.makedirs(csv_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(excel_path))[0]

            xls = pd.ExcelFile(excel_path)
            sheet_names = xls.sheet_names

            csv_paths = []

            for sheet_name in sheet_names:
                safe_sheet_name = "".join(
                    c if c.isalnum() or c in ('-', '_') else '_'
                    for c in sheet_name.lower()
                )
                csv_path = os.path.join(
                    csv_dir,
                    f"{base_name}_{safe_sheet_name}.csv"
                )
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                
                logger.info(f"Successfully converted sheet '{sheet_name}' to CSV: "
                            f"{csv_path}")
                csv_paths.append(csv_path)
            return csv_paths
            
        except Exception as e:
            logger.error(f"Error converting Excel to CSV: {str(e)}")
            raise
    
    def _load_pdf(self, file_path: str) -> List[LoadedUnstructuredDocument]:
        """Load a PDF document."""
        metadata = {
            'source': file_path,
            'type': 'pdf'
        }
        try:
            with open(file_path, 'rb') as file:
                pdf = pypdf.PdfReader(file)
                if len(pdf.pages) == 0:
                    raise ValueError(f"PDF file {file_path} is empty")
                content = []
                total_chars = 0
                
                for i, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text()
                        if text:
                            content.append(text)
                            chars_in_page = len(text)
                            total_chars += chars_in_page
                            metadata[f'page_{i}_length'] = str(chars_in_page)
                
                    except Exception as e:
                        logger.warning("Error reading page {i} of "
                                       f" PDF {file_path}: {str(e)}")
                metadata['total_pages'] = str(len(pdf.pages))
                
                full_text = "\n\n".join(content)
                
                if not full_text.strip():
                    raise ValueError("No text could be extracted from "
                                     f"PDF {file_path}")
                doc = [LoadedUnstructuredDocument(
                    content=full_text, metadata=metadata)]
                return doc
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error reading PDF {file_path}: {str(e)}")
    
    def _load_txt(self, file_path: str) -> List[LoadedUnstructuredDocument]:
        """Load a TXT document."""
        metadata = {
            'source': file_path,
            'type': 'txt'
        }
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if not content.strip():
                    raise ValueError(f"TXT file {file_path} is empty")
                metadata['length'] = str(len(content))
                return [LoadedUnstructuredDocument(
                    content=content, metadata=metadata)]
        except FileNotFoundError:
            raise FileNotFoundError(f"TXT file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error reading TXT file {file_path}: {str(e)}")

    def _load_document(
        self, cfg: DictConfig, path_cfg: DictConfig
    ) -> Union[LoadedUnstructuredDocument,
               List[Union[LoadedUnstructuredDocument, LoadedStructuredDocument]]
               ]:
        """Load a document based on its file type and return its contents with
        metadata.
        
        Args:
            cfg: Hydra configuration object containing settings for document
            loading.
            path_cfg: Configuration object containing the path and related
            settings for the document.
            
        Returns:
            A LoadedUnstructuredDocument or a list of LoadedStructuredDocument
            objects containing the document's content and metadata.
        
        Raises:
            ValueError: If the file format is not supported or if there is an
            issue with the file content.
            FileNotFoundError: If the specified file does not exist.
        """
        file_path = path_cfg.path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self._load_pdf(file_path)
        elif file_ext == '.txt':
            return self._load_txt(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            csv_paths = self.convert_excel_to_csv(
                file_path, cfg.local_doc.csv_dir
            )
            logger.info(f"Converted Excel file {file_path} to CSV: {csv_paths}")
        elif file_ext == '.csv':
            csv_paths = [file_path]
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        # only executes for Excel and CSV files
        loaded_docs = []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            metadata = {
                'source': file_path,
                'type': 'structured',
                'rows': len(df),
                'columns': len(df.columns),
                'rows_threshold': cfg.local_doc.rows_threshold,
                'processing_type': (
                    'full' if len(df) <= cfg.local_doc.rows_threshold
                    else 'chunked'
                )
            }
            loaded_docs.append(
                LoadedStructuredDocument(
                    content=df,
                    metadata=metadata
                )
            )
        return loaded_docs


def load_local_doc(
    cfg: DictConfig
) -> List[Union[LoadedUnstructuredDocument, LoadedStructuredDocument]]:
    """Load documents from local filesystem based on configuration.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        List of loaded documents
    """
    documents = []
    paths = cfg.local_doc.paths
    for path in paths:
        try:
            doc_loader = LocalDocLoader()
            loaded_docs = doc_loader._load_document(cfg, path)

            if isinstance(loaded_docs, list):
                documents.extend(loaded_docs)
            else:
                documents.append(loaded_docs)
            logger.info(f"Successfully loaded document: {path['path']}")
        except Exception as e:
            logger.error(f"Error loading document {path['path']}: {str(e)}")
    logger.info(f"Total {len(documents)} documents loaded.")
    logger.info(f"Docs after loading: {documents}")
    return documents
