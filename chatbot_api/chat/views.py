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

# ✅ New view for edit-profile page
def edit_profile_page(request):
    return render(request, "edit_profile.html")
def reset_password_page(request):
    return render(request, "reset_password.html")

def view_orders_page(request):
    return render(request, "view_orders.html")

def delete_account_page(request):
    return render(request, "delete_account.html")