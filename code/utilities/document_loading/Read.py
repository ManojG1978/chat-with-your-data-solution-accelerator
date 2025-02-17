from typing import List
from .DocumentLoadingBase import DocumentLoadingBase
from ..helpers.AzureFormRecognizerHelper import AzureFormRecognizerClient
from ..common.SourceDocument import SourceDocument


class ReadDocumentLoading(DocumentLoadingBase):
    def __init__(self) -> None:
        super().__init__()

    def load(self, document_url: str) -> List[SourceDocument]:
        """
        Load the document from the specified URL and return a list of SourceDocument objects.

        Parameters:
        - document_url (str): The URL of the document to load.

        Returns:
        - List[SourceDocument]: A list of SourceDocument objects representing the loaded document.
        """
        azure_form_recognizer_client = AzureFormRecognizerClient()
        pages_content = azure_form_recognizer_client.begin_analyze_document_from_url(
            document_url, use_layout=False)
        documents = [
            SourceDocument(
                content=page['page_text'],
                source=document_url,
                page_number=page['page_number'],
                offset=page['offset'],
            )
            for page in pages_content
        ]
        return documents
