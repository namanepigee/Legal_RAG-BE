from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0003_chunk_section_metadata"),
    ]

    operations = [
        migrations.DeleteModel(
            name="DocumentChunk",
        ),
    ]
