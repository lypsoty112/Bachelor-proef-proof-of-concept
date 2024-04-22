from typing import List, Type

from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
import tempfile

from src.components.base.baseComponent import BaseComponent


class PdfProcessor(BaseComponent):
    def __init__(self):
        super().__init__()
        self.loader: Type[PyMuPDFLoader] = PyMuPDFLoader

    def build(self) -> object | None:
        return None

    def run(self, data: UploadFile) -> List[Document]:
        assert data.content_type == "application/pdf", "File must be a PDF"
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(data.file.read())
            temp.close()
            loader = self.loader(temp.name)
            data = loader.load()
            return data

    def post_run(self, data: object) -> object:
        return super().post_run(data)
