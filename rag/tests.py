import unittest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document as LangChainDocument
from pydantic import BaseModel, Field

from rag.service import _generate_node, RagState

class AnswerWithSources(BaseModel):
    answer: str
    used_sources: list[int]

class TestRagService(unittest.TestCase):
    @patch("rag.service._ai_enabled", return_value=True)
    @patch("rag.service._chat_model")
    def test_strict_grounding_and_source_selection(self, mock_chat_model, mock_ai_enabled):
        mock_model_instance = MagicMock()
        mock_chat_model.return_value = mock_model_instance
        
        mock_structured = MagicMock()
        mock_model_instance.with_structured_output.return_value = mock_structured
        
        mock_structured.invoke.return_value = AnswerWithSources(
            answer="This is grounded answer.",
            used_sources=[100]
        )
        
        state = RagState(
            question="What is the test?",
            document_id=1,
            history=[],
            context=[
                LangChainDocument(page_content="Text 1", metadata={"chunk_id": 100, "document_title": "Doc1"}),
                LangChainDocument(page_content="Text 2", metadata={"chunk_id": 101, "document_title": "Doc1"}),
            ],
            answer="",
            validated_sources=[]
        )
        
        new_state = _generate_node(state)
        
        # Validated sources should only contain chunk 100
        self.assertEqual(len(new_state["validated_sources"]), 1)
        self.assertEqual(new_state["validated_sources"][0].metadata["chunk_id"], 100)
        self.assertEqual(new_state["answer"], "This is grounded answer.")

    @patch("rag.service._ai_enabled", return_value=True)
    @patch("rag.service._chat_model")
    def test_invalid_source_id_discarded(self, mock_chat_model, mock_ai_enabled):
        mock_model_instance = MagicMock()
        mock_chat_model.return_value = mock_model_instance
        mock_structured = MagicMock()
        mock_model_instance.with_structured_output.return_value = mock_structured
        
        # Returns an invalid source ID 999
        mock_structured.invoke.return_value = AnswerWithSources(
            answer="Answer.",
            used_sources=[999]
        )
        
        state = RagState(
            question="What is the test?",
            document_id=1,
            history=[],
            context=[
                LangChainDocument(page_content="Text 1", metadata={"chunk_id": 100, "document_title": "Doc1"}),
            ],
            answer="",
            validated_sources=[]
        )
        
        new_state = _generate_node(state)
        
        # Should be empty since 999 is invalid
        self.assertEqual(len(new_state["validated_sources"]), 0)

if __name__ == "__main__":
    unittest.main()
