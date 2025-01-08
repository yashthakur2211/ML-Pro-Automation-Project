# Generated by Django 5.1.4 on 2025-01-06 18:02

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('DataFlowDesk', '0003_dataset_cleaned_file'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.RemoveField(
            model_name='mlmodel',
            name='hyperparameters',
        ),
        migrations.AddField(
            model_name='dataset',
            name='dataset_type',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AddField(
            model_name='dataset',
            name='graphs',
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name='dataset',
            name='target_class',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AddField(
            model_name='dataset',
            name='user',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, related_name='datasets', to=settings.AUTH_USER_MODEL),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='mlmodel',
            name='model_path',
            field=models.FileField(default='', max_length=10000, upload_to='models/'),
            preserve_default=False,
        ),
    ]
