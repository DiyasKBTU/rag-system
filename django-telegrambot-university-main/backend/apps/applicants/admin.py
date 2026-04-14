from django.contrib import admin
from .models import Menu


@admin.register(Menu)
class MenuAdmin(admin.ModelAdmin):
    list_display = ("slug", "title_ru", "title_kk", "title_en", "parent", "order")
    list_filter = ("parent",)
    search_fields = ("slug", "title_ru", "title_kk", "title_en")
    ordering = ("order",)
    prepopulated_fields = {"slug": ("title_ru",)}
    fieldsets = (
        ("Menu", {"fields": ("slug", "parent", "order")}),
        ("Titles", {"fields": ("title_kk", "title_ru", "title_en")}),
        (
            "Contents (leaf only)",
            {"fields": ("content_kk", "content_ru", "content_en")},
        ),
    )