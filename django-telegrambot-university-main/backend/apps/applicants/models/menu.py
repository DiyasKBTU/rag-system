from django.db import models


class Menu(models.Model):
    # Multilingual titles
    title_kk = models.CharField("Title (KK)", max_length=255)
    title_ru = models.CharField("Title (RU)", max_length=255)
    title_en = models.CharField("Title (EN)", max_length=255)

    slug = models.SlugField(unique=True)

    parent = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        related_name="children",
        on_delete=models.CASCADE,
        verbose_name="Parent menu",
    )
    order = models.PositiveIntegerField(default=0)

    # Multilingual content (for leaf menus)
    content_kk = models.TextField("Content (KK)", blank=True)
    content_ru = models.TextField("Content (RU)", blank=True)
    content_en = models.TextField("Content (EN)", blank=True)

    class Meta:
        ordering = ["order"]

    def __str__(self):
        # Default display in admin
        return self.title_ru or self.title_kk or self.title_en

    def get_title(self, lang: str) -> str:
        lang = (lang or "ru").lower()
        if lang == "kk":
            return self.title_kk
        if lang == "en":
            return self.title_en
        return self.title_ru

    def get_content(self, lang: str) -> str:
        lang = (lang or "ru").lower()
        if lang == "kk":
            return self.content_kk
        if lang == "en":
            return self.content_en
        return self.content_ru
