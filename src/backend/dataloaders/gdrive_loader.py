import os
from typing import Optional, Dict
import pandas as pd
from dataclasses import dataclass
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request
from typing import List
import json
from omegaconf import DictConfig
import logging
import io
from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Simple document class for storing content and metadata."""
    content: str
    metadata: Dict
    doc_type: str  # 'pdf', 'sheet', or 'doc'


class GoogleDriveLoader:
    """
    A class for loading data from Google Drive.
    """
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize the loader with optional direct credentials path.
        """
        self.credentials_path = credentials_path
        if not self.credentials_path:
            raise ValueError("No credentials path provided.")
        
        # Verify credentials file exists
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(
                f"Credentials file not found at: {self.credentials_path}")

        self.service_account_email = self._get_service_account_email()
        try:
            self.drive_service = self._initialize_service('drive')
            self.sheets_service = self._initialize_service('sheets')
            self.docs_service = self._initialize_service('docs')
        except Exception as e:
            logger.error(f"Error initializing services: {str(e)}")
            raise

    def _get_service_account_email(self) -> str:
        """Get service account email from credentials file."""
        try:
            with open(self.credentials_path) as f:
                creds = json.load(f)
                return creds.get('client_email', '')
        except Exception as e:
            raise ValueError(f"Error reading credentials file: {str(e)}")

    def _initialize_service(self, service_name: str):
        """
        Initialize a Google service with the appropriate scope.
        
        Args:
            service_name: Name of the service ('sheets', 'drive', or 'docs')
            scope_suffix: The scope suffix (e.g., 'spreadsheets.readonly')
        
        Returns:
            The initialized service
        """
        try:
            service_scopes = {
                'drive': 'https://www.googleapis.com/auth/drive',
                'sheets': 'https://www.googleapis.com/auth/spreadsheets.readonly',
                'docs': 'https://www.googleapis.com/auth/documents.readonly'
            }
            api_versions = {
                'sheets': 'v4',
                'drive': 'v3',
                'docs': 'v1'
            }
            scope = service_scopes.get(service_name)
            version = api_versions.get(service_name)
            if not scope or not version:
                raise ValueError(f"Unsupported service: {service_name}")
            
            cred = service_account.Credentials.from_service_account_file(
                filename=self.credentials_path,
                scopes=[scope]
            )
            # Detailed credential validation
            if not cred:
                raise ValueError("Failed to create credentials")

            if cred.expired:
                logger.info("Credentials expired, attempting refresh...")
                request = Request()
                cred.refresh(request)

            service = build(
                service_name,
                version,
                credentials=cred,
                cache_discovery=False  # Disable file cache
            )
            
            # Test API connection with minimal request
            try:
                if service_name == 'drive':
                    service.files().list(pageSize=1).execute()
                    logger.info("Successfully tested Drive API connection")
                elif service_name == 'sheets':
                    logger.info("Sheets service initialized")
                elif service_name == 'docs':
                    logger.info("Docs service initialized")
            except Exception as e:
                logger.error(f"API test failed for {service_name}: {str(e)}")
                raise ValueError(f"API connection test failed: {str(e)}")
                
            return service
                
        except Exception as e:
            error_msg = f"Failed to initialize Google {service_name} service: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def _get_sheet_data(self, file_id: str) -> Dict[str, pd.DataFrame]:
        """
        Get all sheets and their data from the spreadsheet.
        """
        try:
            # Get spreadsheet metadata to get sheet names
            spreadsheet = self.sheets_service.spreadsheets().get(
                spreadsheetId=file_id
            ).execute()
            
            sheets_data = {}
            
            # Process each sheet
            for sheet in spreadsheet.get('sheets', []):
                sheet_name = sheet['properties']['title']
                logger.info(f"Processing sheet: {sheet_name}")
                
                # Get the sheet's data
                result = self.sheets_service.spreadsheets().values().get(
                    spreadsheetId=file_id,
                    range=sheet_name
                ).execute()
                
                values = result.get('values', [])
                if not values:
                    logger.warning(f"Sheet '{sheet_name}' is empty")
                    continue

                # Convert to DataFrame
                df = pd.DataFrame(values[1:], columns=values[0])
                sheets_data[sheet_name] = df
                logger.info(f"Loaded {len(df)} rows from sheet '{sheet_name}'")
                
            if not sheets_data:
                raise ValueError("No data found in any sheet")
                
            return sheets_data
            
        except Exception as e:
            error_msg = f"Error getting sheet data: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def _get_doc_content(self, doc_id: str) -> List[Document]:
        """Get content from a Google Doc."""
        try:
            doc = self.docs_service.documents().get(documentId=doc_id).execute()
            content = []
            
            # Extract text content from the document
            for element in doc.get('body').get('content'):
                if 'paragraph' in element:
                    for para_element in element.get('paragraph').get('elements'):
                        if 'textRun' in para_element:
                            content.append(para_element.get('textRun').get('content'))
            
            text_content = ''.join(content)
            
            # Create a LangChain Document
            return [Document(
                page_content=text_content,
                metadata={
                    'source': f'gdoc_{doc_id}',
                    'title': doc.get('title', ''),
                    'type': 'google_doc'
                }
            )]
            
        except Exception as e:
            raise Exception(f"Error getting Google Doc content: {str(e)}")

    def _get_pdf_content(self, file_id: str) -> List[Document]:
        """Download and process PDF from Google Drive."""
        try:
            # Download the PDF file
            request = self.drive_service.files().get_media(fileId=file_id)
            file = io.BytesIO()
            downloader = MediaIoBaseDownload(file, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()

            # Reset file pointer to beginning
            file.seek(0)
            
            pdf_reader = PdfReader(file)
            content = []
            for page in pdf_reader.pages:
                content.append(page.extract_text())
                
            # Get PDF metadata
            metadata = {
                'num_pages': len(pdf_reader.pages),
                'pdf_info': pdf_reader.metadata,  # This gets PDF document info
            }
            
            return {
                'content': "\n\n".join(content),
                'metadata': metadata
            }
                
        except Exception as e:
            raise Exception(f"Error getting PDF from Google Drive: {str(e)}")

    def _load_document(self, file_id: str, file_type: str) -> List[Document]:
        """
        Load a single document from Google Drive.
        
        Args:
            file_id: The Google Drive file ID
            file_type: Type of file ('sheet', 'doc', or 'pdf')
            
        Returns:
            List of loaded documents (multiple in case of spreadsheets with multiple sheets)
        """
        documents = []
        
        try:
            # Validate file type
            file_type = file_type.lower()
            if file_type not in ['sheets', 'docs', 'pdf']:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            try:
                file_info = self.drive_service.files().get(
                    fileId=file_id,
                    fields='id, name, mimeType, createdTime, modifiedTime'
                ).execute()
                logger.info(f"Retrieved metadata for file: {file_info.get('name')}")
            except Exception as e:
                raise ValueError(f"Error retrieving file metadata: {str(e)}")
            
            # Common metadata for all document types
            base_metadata = {
                'source_id': file_id,
                'filename': file_info.get('name'),
                'created_time': file_info.get('createdTime'),
                'modified_time': file_info.get('modifiedTime'),
                'mime_type': file_info.get('mimeType')
            }

            if file_type.lower() == 'sheets':
                # Handle spreadsheets
                sheet_data = self._get_sheet_data(file_id)
                for sheet_name, df in sheet_data.items():
                    sheet_metadata = {
                     **base_metadata,
                     'sheet_name': sheet_name,
                     'columns': df.columns.tolist(),
                     'row_count': len(df)
                     }
                    
                    doc = Document(
                        content=df.to_string(),
                        metadata=sheet_metadata,
                        doc_type='sheets'
                    )
                    documents.append(doc)
                    logger.info(f"Loaded sheet '{sheet_name}' with {len(df)} rows")
                       
            elif file_type.lower() == 'docs':
                # Handle Google Docs
                content = self._get_doc_content(file_id)
                doc = Document(
                    content=content,
                    metadata=base_metadata,
                    doc_type='docs'
                )
                documents.append(doc)
                logger.info(f"Loaded Google Doc with {len(content)} characters")
                
            elif file_type.lower() == 'pdf':
                # Handle PDFs
                content = self._get_pdf_content(file_id)
                doc = Document(
                    content=content,
                    metadata=base_metadata,
                    doc_type='pdf'
                )
                documents.append(doc)
                logger.info(f"Loaded PDF with {len(content)} characters")

            return documents
            
        except Exception as e:
            logger.error(f"Error in load_document for {file_id}: {str(e)}")
            raise

    def load_documents(self, cfg: DictConfig) -> List[Document]:
        """
        Load multiple documents from Google Drive based on configuration.
        
        Args:
            cfg: Configuration containing GDRIVE_DOC settings with:
                - FILE_ID: The Google Drive file ID
                - FILE_TYPE: Type of file ('sheet', 'doc', or 'pdf')
            
        Returns:
            List of loaded documents
        """
        documents = []
        for doc_cfg in cfg.grive_doc:
            try:
                docs = self._load_document(
                    file_id=doc_cfg['file_id'],
                    file_type=doc_cfg['file_type']
                )
                documents.extend(docs)
                logger.info(
                    f"Successfully loaded Google Drive document: "
                    f"{doc_cfg['file_id']} ({doc_cfg['file_type']})"
                )
            except Exception as e:
                logger.error(
                    f"Error loading Google Drive document "
                    f"{doc_cfg['file_id']}: {str(e)}"
                )
        return documents
