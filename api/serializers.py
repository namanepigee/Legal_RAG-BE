from rest_framework import serializers

from .models import ChatMessage, Document, DocumentChunk


class DocumentSerializer(serializers.ModelSerializer):
    file = serializers.FileField(write_only=True, required=False)
    chunk_count = serializers.IntegerField(read_only=True)
    message_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Document
        fields = [
            "id",
            "title",
            "content",
            "original_filename",
            "page_count",
            "chunk_count",
            "message_count",
            "created_at",
            "file",
        ]
        read_only_fields = ["id", "original_filename", "page_count", "created_at"]
        extra_kwargs = {
            "title": {"required": False, "allow_blank": True},
            "content": {"required": False, "allow_blank": True},
        }

    def validate(self, attrs):
        if not attrs.get("file") and not attrs.get("content"):
            raise serializers.ValidationError("Upload a file or provide document text.")
        return attrs


class QuerySerializer(serializers.Serializer):
    question = serializers.CharField()
    document_id = serializers.IntegerField(required=False, allow_null=True)


class DocumentChunkSerializer(serializers.ModelSerializer):
    word_count = serializers.SerializerMethodField()
    has_embedding = serializers.SerializerMethodField()

    class Meta:
        model = DocumentChunk
        fields = [
            "id",
            "chunk_index",
            "chapter_title",
            "section_number",
            "section_title",
            "subsection_number",
            "page_start",
            "page_end",
            "word_count",
            "has_embedding",
            "content",
            "created_at",
        ]

    def get_word_count(self, obj):
        return len(obj.content.split())

    def get_has_embedding(self, obj):
        return bool(obj.embedding)


class ChatMessageSerializer(serializers.ModelSerializer):
    sources = serializers.SerializerMethodField()

    class Meta:
        model = ChatMessage
        fields = ["id", "document", "role", "content", "sources", "created_at"]

    def get_sources(self, obj):
        repaired_sources = []
        for source in obj.sources or []:
            source = dict(source)
            pages = source.get("pages")
            if pages and pages != "not available":
                repaired_sources.append(source)
                continue

            chunk = None
            chunk_id = source.get("chunk_id")
            if chunk_id:
                chunk = DocumentChunk.objects.filter(id=chunk_id).first()

            if not chunk and obj.document_id:
                chunk = (
                    DocumentChunk.objects.filter(
                        document_id=obj.document_id,
                        section_title=source.get("section_title") or "",
                    )
                    .order_by("chunk_index")
                    .first()
                )

            if chunk and chunk.page_start:
                source["chunk_id"] = chunk.id
                source["chunk_index"] = chunk.chunk_index
                source["pages"] = (
                    f"{chunk.page_start}-{chunk.page_end}"
                    if chunk.page_end and chunk.page_end != chunk.page_start
                    else str(chunk.page_start)
                )
                source["preview"] = chunk.content[:600]

            repaired_sources.append(source)
        return repaired_sources
