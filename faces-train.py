import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #location of path
image_dir = os.path.join(BASE_DIR, "image") # looks for the image folder

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}  #creates dictionary
y_labels = []   #creates lists 
x_train = []

for root, dirs, files in os.walk(image_dir): #finds each file path
	for file in files:
		if file.endswith("png") or file.endswith("jpg"): 
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower()  #removes blank spaces from files

			if not label in label_ids: #creates id with persons name attached
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
   
			pil_image = Image.open(path).convert("L") # grayscale
			size = (550, 550) #resizes image
			final_image = pil_image.resize(size, Image.LANCZOS) #image scaling
			image_array = np.array(final_image, "uint8") #convert to numpy array
			faces = face_cascade.detectMultiScale(image_array) #detects face in image

			for (x,y,w,h) in faces: 
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)

with open("labels.pickle", 'wb') as f: #using pickle to save label IDs
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels)) #trains the recognizer
recognizer.save("trainner.yml")