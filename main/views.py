from django.shortcuts import render
from shelf_product_script import main as shelf_product_script
import os
# Create your views here.

def index(request):
    return render(request, 'main/index.html')


def scene(request):
    return render(request, 'main/scene.html')


def all_scene(request):
    return render (request, 'main/all-scene.html')
    

def camera_gslv2(request):
    return render(request, 'main/camera-gslv2.html')


def test(request):
    shelf_product_script.run('media/video.mp4', 'media/output.mp4')
    return render(request, 'main/test.html')