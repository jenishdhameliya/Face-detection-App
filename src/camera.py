from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request
import numpy as np
import pickle
from django.conf import settings
import time
import datetime
from .models import Members,Log

class VideoCamera(object):
	def __init__(self):
		self.curr_time = datetime.datetime.now()
		self.video = cv2.VideoCapture(0)
		curr_path = settings.BASE_DIR + "\\src\\"

		print("Loading face detection model")
		self.proto_path = os.path.join(curr_path, 'model', 'deploy.prototxt')
		self.model_path = os.path.join(curr_path, 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
		self.face_detector = cv2.dnn.readNetFromCaffe(prototxt=self.proto_path, caffeModel=self.model_path)

		print("Loading face recognition model")
		self.recognition_model = os.path.join(curr_path, 'model', 'openface_nn4.small2.v1.t7')
		self.face_recognizer = cv2.dnn.readNetFromTorch(model=self.recognition_model)

		self.recognizer = pickle.loads(open('src/recognizer.pickle', "rb").read())
		self.le = pickle.loads(open('src/le.pickle', "rb").read())


	def __del__(self):
		self.video.release()

	def get_frame(self):	
		ret, frame = self.video.read()
		frame = imutils.resize(frame, width=600)
		frame = cv2.flip(frame,1)
		(h, w) = frame.shape[:2]

		self.image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

		self.face_detector.setInput(self.image_blob)
		self.face_detections = self.face_detector.forward()
		for i in range(0, self.face_detections.shape[2]):
			confidence = self.face_detections[0, 0, i, 2]

			if confidence >= 0.5:
				box = self.face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				face = frame[startY:endY, startX:endX]

				(fH, fW) = face.shape[:2]

				face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), True, False)

				self.face_recognizer.setInput(face_blob)
				vec = self.face_recognizer.forward()

				preds = self.recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = self.le.classes_[j]

				text = "{:.2f}".format(proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				qs = Members.objects.filter(name = name)
				if (name == "unknown" or len(qs) == 0):
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
					cv2.putText(frame,"unknown",(20,20), cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
					cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
					usr = Members.objects.filter(name = "unknown")
					if(len(usr) > 0):
						usr = usr[0]
					entries = Log.objects.filter(member=usr).order_by('-entry')
					if(len(entries) == 0):
						entry = Log.objects.create(entry=datetime.datetime.now(),member=usr)
						entry.save()
					else:
						entries = entries[0].entry
						curr = datetime.datetime.now()
						if(entries.second + 5 <= curr.second):
							entry = Log.objects.create(entry=curr,member=usr)
							entry.save()
				else:
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
					cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
					cv2.putText(frame,name,(20,20), cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,0),2)
					cv2.putText(frame,"age: " + str(qs[0].age),(20,50), cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,0),2)
					usr = Members.objects.filter(name = name)[0]
					entries = Log.objects.filter(member=usr).order_by('-entry')
					if(len(entries) == 0):
						entry = Log.objects.create(entry=datetime.datetime.now(),member=usr)
						entry.save()
					else:
						entries = entries[0].entry
						curr = datetime.datetime.now()
						print(entries.second,curr.second)
						
						if(entries.hour == curr.hour and entries.day == curr.day and entries.minute == curr.minute and (entries.second + 5) >= curr.second):
							pass
						else:
							entry = Log.objects.create(entry=curr,member=usr)
							entry.save()



		ret, jpeg = cv2.imencode('.jpg', frame)
		return jpeg.tobytes()