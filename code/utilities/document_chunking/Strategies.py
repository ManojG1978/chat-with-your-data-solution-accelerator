from enum import Enum


class ChunkingStrategy(Enum):
    """
    Enumeration representing different strategies for chunking a document.
    """

    LAYOUT = 'layout'
    PAGE = 'page'
    FIXED_SIZE_OVERLAP = 'fixed_size_overlap'
    PARAGRAPH = 'paragraph'


def get_document_chunker(chunking_strategy: str):
    if chunking_strategy == ChunkingStrategy.LAYOUT.value:
        from .Layout import LayoutDocumentChunking
        return LayoutDocumentChunking()
    elif chunking_strategy == ChunkingStrategy.PAGE.value:
        from .Page import PageDocumentChunking
        return PageDocumentChunking()
    elif chunking_strategy == ChunkingStrategy.FIXED_SIZE_OVERLAP.value:
        from .FixedSizeOverlap import FixedSizeOverlapDocumentChunking
        return FixedSizeOverlapDocumentChunking()
    elif chunking_strategy == ChunkingStrategy.PARAGRAPH.value:
        from .Paragraph import ParagraphDocumentChunking
        return ParagraphDocumentChunking()
    else:
        raise Exception(f"Unknown chunking strategy: {chunking_strategy}")


class ChunkingSettings:
    """
    Represents the settings for document chunking.

    Args:
        chunking (dict): A dictionary containing the chunking settings.
            - strategy (str): The strategy for chunking.
            - size (int): The size of each chunk.
            - overlap (int): The overlap between consecutive chunks.
    """

    def __init__(self, chunking: dict):
        self.chunking_strategy = ChunkingStrategy(chunking['strategy'])
        self.chunk_size = chunking['size']
        self.chunk_overlap = chunking['overlap']
