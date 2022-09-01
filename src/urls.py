from django.urls import path, include
from .views import video_feed,index,AddMemberPage, create_dataset,off,download
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('',index, name='index'),
    path('offline',off,name="off"),
    path('video_feed',video_feed, name='video_feed'),
    path("add-member",AddMemberPage,name="add_member"),
    path('capture/<str:name>',create_dataset,name='capture1'),
    path('download',download,name="download"),
   
]
urlpatterns += static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)
