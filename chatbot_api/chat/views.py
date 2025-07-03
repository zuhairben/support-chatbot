from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .logic import handle_user_input

class ChatbotPageView(APIView):
    def get(self, request):
        return render(request, "chatbot.html")

class ChatbotAPIView(APIView):
    def post(self, request):
        user_input = request.data.get("message", "")
        result = handle_user_input(user_input)
        return Response(result)
