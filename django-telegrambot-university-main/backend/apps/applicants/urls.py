from django.urls import path
from .views import MenuListAPIView, AskFromDocAPIView

urlpatterns = [
    path("menu/", MenuListAPIView.as_view()),
    path("ask/", AskFromDocAPIView.as_view()),
]
