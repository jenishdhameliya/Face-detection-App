from django.shortcuts import render,redirect
from .forms import Memberform
from django.http.response import StreamingHttpResponse,HttpResponse
from .camera import VideoCamera
import cv2
import os
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from PIL import Image
from django.conf import settings
from time import sleep
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
from .models import Members,Log
import imutils
import datetime
from sklearn.preprocessing import LabelEncoder
import xlwt

# Create your views here.
def index(request):     
    peps = Members.objects.all()
    return render(request, 'base.html',{"member":peps})

def off(request):
    data = Log.objects.all()
    return render(request,'base.html',{"state":"off","data":data})
    
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                    content_type='multipart/x-mixed-replace; boundary=frame')

def AddMemberPage(request):
    if(request.method == "GET"):
        form = Memberform()
        return render(request,"addMember.html",{'form':form})
    else:
        #print(request.POST)
        name = Members.objects.create(name=request.POST['name'],age=request.POST['age'])
        form = Memberform()
        #print(form)
        return render(request,"addMember.html",{'gathered':True,"name":name})

def create_dataset(request,name):
    member = name
    cam = cv2.VideoCapture(0)
    parent_dir = settings.BASE_DIR + "\\src\\dataset\\"
    path = os.path.join(parent_dir,name)
    os.mkdir(path)
    total = 0
    while(True):
        ret,img = cam.read()
        total += 1
        sleep(1)
        if(total > 10):
            break
        cv2.imwrite(parent_dir+"\\" + name +"\\"+str(total)+".jpg",img)
    print("Loading face detection model")
    curr_path = os.getcwd() + "\\src\\"
    proto_path = os.path.join(curr_path, 'model', 'deploy.prototxt')
    model_path = os.path.join(curr_path, 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
    face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

    print("Loading face recognition model")
    recognition_model = os.path.join(curr_path, 'model', 'openface_nn4.small2.v1.t7')
    face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

    data_base_path = os.path.join(curr_path, 'dataset')

    filenames = []
    for path, subdirs, files in os.walk(data_base_path):
        for name in files:
            filenames.append(os.path.join(path, name))

    face_embeddings = []
    face_names = []

    for (i, filename) in enumerate(filenames):
        print("Processing image {}".format(filename))

        image = cv2.imread(filename)
        image = imutils.resize(image, width=600)

        (h, w) = image.shape[:2]

        image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

        face_detector.setInput(image_blob)
        face_detections = face_detector.forward()

        i = np.argmax(face_detections[0, 0, :, 2])
        confidence = face_detections[0, 0, i, 2]

        if confidence >= 0.5:

            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]

            face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0), True, False)

            face_recognizer.setInput(face_blob)
            face_recognitions = face_recognizer.forward()

            name = filename.split(os.path.sep)[-2]

            face_embeddings.append(face_recognitions.flatten())
            face_names.append(name)


    data = {"embeddings": face_embeddings, "names": face_names}

    le = LabelEncoder()
    labels = le.fit_transform((data["names"]))

    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    f = open('src/recognizer.pickle', "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    f = open("src/le.pickle", "wb")
    f.write(pickle.dumps(le))
    f.close()

    return redirect('/')
        
        
def download(request):
    response = HttpResponse(content_type='application/ms-excel')
    response ['Content-Disposition'] = 'attachment; filename=entries' + \
        str(datetime.datetime.now()) + '.xls'

    wb = xlwt.Workbook(encoding='utf-8')
    ws = wb.add_sheet("Entries")
    row_num = 0
    font_style = xlwt.XFStyle() 
    font_style.font.bold = True
    columns = ["Person","Date","Time"]

    for col_num in range(len(columns)):
        ws.write(row_num,col_num,columns[col_num],font_style)

    font_style = xlwt.XFStyle()
    rows = Log.objects.all()

    for row in rows:
        row_num += 1
        ws.write(row_num,0,row.member.name,font_style)
        
        ws.write(row_num,1,str(row.entry.date()),font_style)
        ws.write(row_num,2,str(row.entry.time()),font_style)

    wb.save(response)
    return response