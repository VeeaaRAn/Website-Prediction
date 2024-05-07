from django.urls import path
from . import views
app_name='feed'

urlpatterns = [
    path('predict/', views.PredictImage.as_view(), name='predict'),
    path('all_post/',views.HomePageView.as_view(),name='index'),
    path('detail/<int:pk>/',views.PostDetailView.as_view(),name='detail'),
    path('all/',views.all_posts,name='some'),
    path('disease',views.Diseaseview.as_view(),name='diseasepage'),
    path('',views.MainPage.as_view(),name='main'),
    path('hello/', views.hello_view, name='hello'),
    path('retina/',views.predict_image,name='predict_image'),
    path('retina/<int:pk>/',views.PostDetailView.as_view(),name='detail'),


]
