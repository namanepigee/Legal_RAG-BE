from django.urls import path

from .views import ChatHistoryView, DocumentChunksView, DocumentListCreateView, HealthView, QueryView

urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("documents/", DocumentListCreateView.as_view(), name="documents"),
    path("documents/<int:document_id>/chunks/", DocumentChunksView.as_view(), name="document-chunks"),
    path("documents/<int:document_id>/messages/", ChatHistoryView.as_view(), name="document-messages"),
    path("query/", QueryView.as_view(), name="query"),
]
