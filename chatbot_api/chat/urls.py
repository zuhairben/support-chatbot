from django.urls import path
from .views import (
    ChatbotAPIView, ChatbotPageView,
    edit_profile_page, reset_password_page,
    view_orders_page, delete_account_page
)

urlpatterns = [
    path("", ChatbotPageView.as_view()),               # Chat UI
    path("chat/", ChatbotAPIView.as_view()),           # API
    path("edit-profile/", edit_profile_page),          # Edit Profile
    path("reset-password/", reset_password_page),      # 🔐 Reset Password
    path("orders/", view_orders_page),                 # 📦 View Orders
    path("delete-account/", delete_account_page),      # ❌ Delete Account
]
