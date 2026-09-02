from rest_framework import serializers

from .models import ChatMessage, Document


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
            "content": {"required": False, "allow_blank": True, "write_only": True},
        }

    def validate(self, attrs):
        if not attrs.get("file") and not attrs.get("content"):
            raise serializers.ValidationError("Upload a file or provide document text.")
        return attrs


class QuerySerializer(serializers.Serializer):
    question = serializers.CharField()
    document_id = serializers.IntegerField(required=False, allow_null=True)


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ["id", "document", "role", "content", "sources", "created_at"]
