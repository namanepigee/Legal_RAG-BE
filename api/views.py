import logging
import os
from threading import Thread

from rest_framework import generics, status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import ChatMessage, Document
from rag.service import answer_question, extract_uploaded_file, index_document, list_document_chunks
from .serializers import ChatMessageSerializer, DocumentSerializer, QuerySerializer

logger = logging.getLogger(__name__)


def index_document_async(document_id, page_texts):
    try:
        document = Document.objects.get(id=document_id)
        document._page_texts = page_texts
        index_document(document)
    except Exception:
        logger.exception("Background indexing failed for document_id=%s", document_id)


class HealthView(APIView):
    def get(self, request):
        provider = os.getenv("AI_PROVIDER", "").lower() or "openai"
        return Response(
            {
                "status": "ok",
                "ai": {
                    "provider": provider,
                    "aws_region": os.getenv("AWS_REGION", ""),
                    "aws_access_key_present": bool(os.getenv("AWS_ACCESS_KEY_ID")),
                    "aws_secret_key_present": bool(os.getenv("AWS_SECRET_ACCESS_KEY")),
                    "bedrock_chat_model": os.getenv("AI_MODEL", ""),
                    "bedrock_embedding_model": os.getenv("BEDROCK_EMBEDDING_MODEL", ""),
                    "openai_key_present": bool(os.getenv("OPENAI_API_KEY")),
                    "openai_chat_model": os.getenv("OPENAI_CHAT_MODEL", ""),
                    "openai_embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", ""),
                    "gemini_key_present": bool(os.getenv("GEMINI_API_KEY")),
                    "gemini_chat_model": os.getenv("GEMINI_CHAT_MODEL", ""),
                    "gemini_embedding_model": os.getenv("GEMINI_EMBEDDING_MODEL", ""),
                },
                "vector_store": {
                    "provider": os.getenv("VECTOR_STORE", "qdrant"),
                    "qdrant_url_present": bool(os.getenv("QDRANT_URL")),
                    "qdrant_api_key_present": bool(os.getenv("QDRANT_API_KEY")),
                    "qdrant_collection": os.getenv("QDRANT_COLLECTION", "legal_document_chunks"),
                },
            }
        )


class DocumentListCreateView(generics.ListCreateAPIView):
    serializer_class = DocumentSerializer
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    def get_queryset(self):
        return Document.objects.order_by("-created_at")

    def perform_create(self, serializer):
        upload = serializer.validated_data.pop("file", None)
        title = serializer.validated_data.get("title") or (upload.name if upload else "Untitled legal document")
        if upload:
            extracted = extract_uploaded_file(upload)
            document = serializer.save(
                title=title,
                content=extracted["content"],
                original_filename=upload.name,
                page_count=extracted["page_count"],
            )
            document._page_texts = extracted["pages"]
        else:
            document = serializer.save(title=title)
            document._page_texts = [{"page": None, "text": document.content}]
        Thread(target=index_document_async, args=(document.id, document._page_texts), daemon=True).start()


class QueryView(APIView):
    def post(self, request):
        serializer = QuerySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        question = serializer.validated_data["question"]
        document_id = serializer.validated_data.get("document_id")
        document = Document.objects.filter(id=document_id).first() if document_id else None
        if document_id and not document:
            return Response({"detail": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        history_qs = ChatMessage.objects.filter(document=document).order_by("-created_at")[:10]
        history_msgs = [{"role": msg.role, "content": msg.content} for msg in reversed(history_qs)]
        
        ChatMessage.objects.create(document=document, role="user", content=question)
        try:
            result = answer_question(question, document_id=document_id, history=history_msgs)
        except Exception:
            logger.exception("Query failed for document_id=%s", document_id)
            return Response(
                {
                    "detail": "Query failed on the backend. Check the Django server terminal logs for the full error.",
                    "answer": "The backend could not complete this query. Please check the Django server terminal logs.",
                    "sources": [],
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        ChatMessage.objects.create(
            document=document,
            role="assistant",
            content=result["answer"],
            sources=result["sources"],
        )
        return Response(result, status=status.HTTP_200_OK)


class DocumentChunksView(APIView):
    def get(self, request, document_id):
        if not Document.objects.filter(id=document_id).exists():
            return Response({"detail": "Document not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(list_document_chunks(document_id), status=status.HTTP_200_OK)


class ChatHistoryView(generics.ListAPIView):
    serializer_class = ChatMessageSerializer

    def get_queryset(self):
        return ChatMessage.objects.filter(document_id=self.kwargs["document_id"]).order_by("created_at")
