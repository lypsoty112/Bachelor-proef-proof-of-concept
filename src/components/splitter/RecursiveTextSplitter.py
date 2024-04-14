from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

from src.components.base.baseComponent import BaseComponent


class RecursiveCharacterTextSplitterComponent(BaseComponent):
    def __init__(self, splitter_params: dict = None) -> None:
        super().__init__()
        self.splitter: RecursiveCharacterTextSplitter | None = None
        self.enc = None

        if splitter_params is None:
            splitter_params = {}

        default_params = {
            "separators": ["\n", ".", "!", "?"],
            "keep_separator": True,
            "chunk_size": 512,
            "length_function": self._length_function,
        }

        for key, value in default_params.items():
            if key not in splitter_params:
                splitter_params[key] = value

        self._params = splitter_params

    def build(self) -> object | None:
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.splitter = RecursiveCharacterTextSplitter(
            **self._params
        )
        return None

    def run(self, data: str | Document | List[str | Document]) -> list[Document]:
        if not isinstance(data, list):
            data = [data]

        for i, item in enumerate(data):
            if not isinstance(item, Document):
                data[i] = Document(
                    page_content=item,
                    metadata={},
                )

        return self.splitter.split_documents(data)

    def post_run(self, data: object) -> object:
        return data

    def _length_function(self, text: str) -> int:
        return len(self.enc.encode(text))
