from django.shortcuts import render

# Create your views here.

def index(request):
    return render(request, 'main/index.html')


def scene(request):
    return render(request, 'main/scene.html')


def all_scene(request):
    return render (request, 'main/all-scene.html')
    

def camera_gslv2(request):
    return render(request, 'main/camera-gslv2.html')