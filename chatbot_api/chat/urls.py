from django.urls import path
from .views import ChatbotAPIView, ChatbotPageView

urlpatterns = [
    path("", ChatbotPageView.as_view()),
    path("chat/", ChatbotAPIView.as_view()),
]
