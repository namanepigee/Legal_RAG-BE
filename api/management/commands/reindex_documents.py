from django.core.management.base import BaseCommand

from api.models import Document
from rag.service import index_document


class Command(BaseCommand):
    help = "Rebuild document chunks and vector-store entries for existing documents."

    def add_arguments(self, parser):
        parser.add_argument("--document-id", type=int, help="Only reindex one document ID.")

    def handle(self, *args, **options):
        queryset = Document.objects.order_by("id")
        if options.get("document_id"):
            queryset = queryset.filter(id=options["document_id"])

        count = queryset.count()
        if not count:
            self.stdout.write(self.style.WARNING("No documents found to reindex."))
            return

        for index, document in enumerate(queryset, start=1):
            self.stdout.write(f"[{index}/{count}] Reindexing document {document.id}: {document.title}")
            index_document(document)

        self.stdout.write(self.style.SUCCESS(f"Reindexed {count} document(s)."))
