from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('scene/', views.scene, name='scene'),
    path('all-scene', views.all_scene, name='all_scene'),
    path('camera-gslv2', views.camera_gslv2, name='camera_gslv2'),
    path('test', views.test, name='test'),
]