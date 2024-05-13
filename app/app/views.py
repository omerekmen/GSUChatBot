# views.py
from django.shortcuts import render
from django.http import JsonResponse
from api.api import ChatBot

def index(request):
    return render(request, 'index.html')

def get_response(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        if user_input:
            bot = ChatBot()
            results = bot.query_search(user_input)
            response = bot.gsu_chatbot(results[0])
            return JsonResponse({'response': response})
    return JsonResponse({'response': 'Invalid request method or missing user input.'}, status=400)
