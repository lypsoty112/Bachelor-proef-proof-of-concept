import json
from typing import Any, Dict, List
from src.chains.baseChain import BaseChain
from src.chains.chatChain import ChatChain
from src.chains.classifierChain import ClassifierChain
from src.chains.contentGeneratorChain import ContentGeneratorChain
from src.chains.testCorrectorChain import TestCorrectorChain
from src.models.chains.chat import ChatInput, ChatOutput
from src.models.chains.classifier import NextStep
from src.models.main import Message
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import loguru


class TeacherChain(BaseChain):
    def __init__(self, course_contents: List[Dict[str, Any]], user_preferences: List[str]):
        super().__init__()
        self._course = course_contents
        self._inner_contents = []
        self._user_preferences = user_preferences

        self._chunk_index = 0
        self._inner_chunk_index = 0

        self._contentGenerator = ContentGeneratorChain()
        self._chatChain = ChatChain()
        self._classifierChain = ClassifierChain()
        self._testCorrectorChain = TestCorrectorChain()

        self._text_splitter = SemanticChunker(OpenAIEmbeddings())
        self._logger = loguru.logger

    def build(self) -> None:
        # Build the chains and set the indexes
        self._contentGenerator.build()
        self._chatChain.build()
        self._classifierChain.build()
        self._testCorrectorChain.build()

        self._chunk_index = 0
        self._inner_chunk_index = 0

    def starting_message(self) -> ChatOutput:
        self._inner_chunk_index = 0
        self._chunk_index = 0

        self.update_inner_contents()
        generated_content = self._contentGenerator.run({
            "course_content": self._inner_contents[self._inner_chunk_index], "employee_needs": self._user_preferences,
        })

        return ChatOutput(response=Message(role="ai", content=generated_content), metadata={"type": "content"})

    def run(self, data: ChatInput) -> ChatOutput:
        # Analyse the input
        self._logger.info(f"Current state: {self._chunk_index}, {self._inner_chunk_index}")
        message = data.messages[-1].content

        classification = self._classifierChain.run({
            "message": message, "chat_history": data.messages[:-1],
        })

        self._logger.info(f"Classification: {classification}")

        if classification.next_step == NextStep.CONTINUE_COURSE:
            # Check if there's a next inner chunk
            if self._inner_chunk_index < len(self._inner_contents) - 1:
                # If there's one, get the next inner chunk
                self._inner_chunk_index += 1
                generated_content = self._contentGenerator.run({
                    "course_content": self._inner_contents[self._inner_chunk_index], "employee_needs": self._user_preferences, "previous_interaction": data.messages,
                })
                return ChatOutput(response=Message(role="ai", content=generated_content), metadata=self.generate_metadata("content"))
            else:
                # Update the inner contents
                self._chunk_index += 1
                self.update_inner_contents()
                self._inner_chunk_index = 0
                generated_content = self._contentGenerator.run({
                    "course_content": self._inner_contents[self._inner_chunk_index], "employee_needs": self._user_preferences, "previous_interaction": data.messages,
                })

                return ChatOutput(response=Message(role="ai", content=generated_content), metadata=self.generate_metadata("content"))

        elif classification.next_step == NextStep.ASK_FOR_MORE_INFORMATION:
            # Ask for more information
            return self._chatChain.run({
                "messages": data.messages.extend([Message(role="system", content="There's not enough information to answer the user's question. Ask for more information.")]),
                "metadata": self.generate_metadata("content", data.metadata)
            })

        elif classification.next_step == NextStep.RESPOND_TO_QUESTION:
            # Respond to the question
            inner_chunk = self._inner_contents[self._inner_chunk_index]

            copied_messages = data.messages.copy()

            metadata = {
                **data.metadata, "course_material": inner_chunk,
            }

            return self._chatChain.run({
                "messages": copied_messages, "metadata": metadata,
            })

    def update_inner_contents(self) -> None:
        # Get the current chunk idx
        chunk_idx = self._chunk_index
        # Get the current chunk
        chunk = self._course[chunk_idx]

        self._inner_contents = self._text_splitter.split_text(chunk["chunk"]["page_content"])

    def generate_metadata(self, message_type: str, existing_metadata: Dict[str, Any] | None = None, **args) -> Dict[str, Any]:
        if existing_metadata is None:
            metadata = {}
        else:
            metadata = existing_metadata.copy()
        metadata["type"] = message_type
        metadata["chunk_index"] = self._chunk_index
        metadata["inner_chunk_index"] = self._inner_chunk_index
        metadata["employee_needs"] = self._user_preferences
        metadata["course_content"] = self._inner_contents[self._inner_chunk_index]
        return {
            **metadata, **args,
        }

    def pre_run(self, data: ChatInput) -> dict:
        pass

    def post_run(self, data: any) -> ChatOutput:
        pass
