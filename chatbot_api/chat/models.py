from django.db import models

class Intent(models.Model):
    name = models.CharField(max_length=100)
    response = models.TextField()
    url = models.CharField(max_length=200, blank=True)

    def __str__(self):
        return self.name

class IntentExample(models.Model):
    intent = models.ForeignKey(Intent, related_name='examples', on_delete=models.CASCADE)
    text = models.CharField(max_length=200)

    def __str__(self):
        return f"{self.intent.name} example: {self.text}"

class IntentKeyword(models.Model):
    intent = models.ForeignKey(Intent, related_name='keywords', on_delete=models.CASCADE)
    word = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.intent.name} keyword: {self.word}"
