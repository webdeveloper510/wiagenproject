# Generated by Django 4.1.7 on 2023-06-09 10:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('authapp', '0007_delete_user_label_questionandanswr_user_id'),
    ]

    operations = [
        migrations.AddField(
            model_name='topic',
            name='user_id',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
    ]
