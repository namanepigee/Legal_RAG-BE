from django.db import models


class Document(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    original_filename = models.CharField(max_length=255, blank=True)
    page_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class DocumentChunk(models.Model):
    document = models.ForeignKey(Document, related_name="chunks", on_delete=models.CASCADE)
    content = models.TextField()
    chapter_title = models.CharField(max_length=255, blank=True)
    section_number = models.CharField(max_length=20, blank=True)
    section_title = models.CharField(max_length=500, blank=True)
    subsection_number = models.CharField(max_length=20, blank=True)
    page_start = models.PositiveIntegerField(null=True, blank=True)
    page_end = models.PositiveIntegerField(null=True, blank=True)
    chunk_index = models.PositiveIntegerField(default=0)
    embedding = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.document.title} chunk {self.pk}"


class ChatMessage(models.Model):
    document = models.ForeignKey(Document, related_name="messages", on_delete=models.CASCADE, null=True, blank=True)
    role = models.CharField(max_length=20)
    content = models.TextField()
    sources = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.role}: {self.content[:80]}"
