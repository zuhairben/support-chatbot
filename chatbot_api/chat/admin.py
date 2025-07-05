from django.contrib import admin
from .models import Intent, IntentExample, IntentKeyword

class IntentExampleInline(admin.TabularInline):
    model = IntentExample
    extra = 1

class IntentKeywordInline(admin.TabularInline):
    model = IntentKeyword
    extra = 1

class IntentAdmin(admin.ModelAdmin):
    list_display = ("name", "url")
    inlines = [IntentExampleInline, IntentKeywordInline]

admin.site.register(Intent, IntentAdmin)
