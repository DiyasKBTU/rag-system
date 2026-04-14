from rest_framework import serializers
from .models import Menu


class MenuSerializer(serializers.ModelSerializer):
    title = serializers.SerializerMethodField()
    content = serializers.SerializerMethodField()
    children = serializers.SerializerMethodField()

    class Meta:
        model = Menu
        fields = ("id", "title", "slug", "content", "children")

    def _lang(self) -> str:
        return (self.context.get("lang") or "ru").lower()

    def get_title(self, obj: Menu) -> str:
        return obj.get_title(self._lang())

    def get_content(self, obj: Menu) -> str:
        return obj.get_content(self._lang())

    def get_children(self, obj):
        qs = obj.children.all()
        return MenuSerializer(qs, many=True, context=self.context).data
