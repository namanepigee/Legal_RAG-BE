import logging

from rest_framework import generics, status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView
from django.db.models import Count

from .models import ChatMessage, Document, DocumentChunk
from rag.service import answer_question, extract_uploaded_file, index_document
from .serializers import ChatMessageSerializer, DocumentChunkSerializer, DocumentSerializer, QuerySerializer

logger = logging.getLogger(__name__)


class HealthView(APIView):
    def get(self, request):
        return Response({"status": "ok"})


class DocumentListCreateView(generics.ListCreateAPIView):
    serializer_class = DocumentSerializer
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    def get_queryset(self):
        return Document.objects.annotate(
            chunk_count=Count("chunks", distinct=True),
            message_count=Count("messages", distinct=True),
        ).order_by("-created_at")

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
        index_document(document)


class QueryView(APIView):
    def post(self, request):
        serializer = QuerySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        question = serializer.validated_data["question"]
        document_id = serializer.validated_data.get("document_id")
        document = Document.objects.filter(id=document_id).first() if document_id else None
        if document_id and not document:
            return Response({"detail": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        ChatMessage.objects.create(document=document, role="user", content=question)
        try:
            result = answer_question(question, document_id=document_id)
        except Exception:
            logger.exception("Query failed for document_id=%s", document_id)
            return Response(
                {
                    "detail": "Query failed on the backend. Check Render logs for the full error.",
                    "answer": "The backend could not complete this query. Please check Render environment variables and logs.",
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


class DocumentChunksView(generics.ListAPIView):
    serializer_class = DocumentChunkSerializer

    def get_queryset(self):
        return DocumentChunk.objects.filter(document_id=self.kwargs["document_id"]).order_by("chunk_index")


class ChatHistoryView(generics.ListAPIView):
    serializer_class = ChatMessageSerializer

    def get_queryset(self):
        return ChatMessage.objects.filter(document_id=self.kwargs["document_id"]).order_by("created_at")
