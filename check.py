#Importings
import os
import sys
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

#Finding image names
os.chdir('/home/vatsalbabel/Downloads/frontalFace10/images')
images = os.popen('ls').read().split('\n')
images.pop(-1)

#Loading the cascade
face_cascade = cv2.CascadeClassifier('/home/vatsalbabel/Downloads/frontalFace10/haarcascade_frontalface_alt.xml')

#Creating feature and labels set
faces = []
labels = []
for img in images:
	img_path = '/home/vatsalbabel/Downloads/frontalFace10/images/' + img
	image = cv2.imread(img_path, 0)
	image = cv2.resize(image, (200, 200))
	face = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
	if not len(face)==0:
		(x, y, w, h) = face[0]
	faces.append(image[y:y+w, x:x+h])
	labels.append(img.split()[0])

#Loading LBPH Model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Encoding the categorial label values
labels = np.array(labels)
le = LabelEncoder()
labels = le.fit_transform(labels)

#Training
face_recognizer.train(faces, labels)

#Predicting based on the test path of the image
img_path = sys.argv[1]
org_image = cv2.imread(img_path)
image = cv2.resize(org_image, (200, 200))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
if len(face)==0:
	print('No face found')
else:
	(x, y, w, h) = face[0]
	predicted = face_recognizer.predict(image[y:y+w, x:x+h])
	text = le.inverse_transform(predicted[0])
	cv2.putText(org_image, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
	cv2.imshow('predicted', org_image)
	if cv2.waitKey(0) & 0xFF==ord('q'):
		cv2.destroyAllWindows()
