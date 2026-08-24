from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0002_chatmessage"),
    ]

    operations = [
        migrations.AddField(
            model_name="documentchunk",
            name="chapter_title",
            field=models.CharField(blank=True, max_length=255),
        ),
        migrations.AddField(
            model_name="documentchunk",
            name="section_number",
            field=models.CharField(blank=True, max_length=20),
        ),
        migrations.AddField(
            model_name="documentchunk",
            name="subsection_number",
            field=models.CharField(blank=True, max_length=20),
        ),
    ]
