'''
PyPower Projects
Mask Detection Using Machine Learning
'''

#USAGE : python predict_video.py

from tensorflow.keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('./face.xml')
classifier = load_model('./testemodel.h5')

class_labels = ['Com Mascara','Sem Mascara']
start_point = (15, 15)
end_point = (400, 80) 
thickness = -1
scale_factor = 1.3

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = vc.read()
    labels = []
    
    faces = face_classifier.detectMultiScale(frame, scale_factor, 5)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Rosto", (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    gray = cv2.resize(frame,(224,224))

    roi = np.expand_dims(gray,axis=0)

    # make a prediction on the ROI, then lookup the class

    (mask, withoutMask) = model.predict(roi)[0]
    label = "Com Mascara" if mask > withoutMask else "Sem Mascara"
    
    preds = classifier.predict(roi)[0]
    label=class_labels[preds.argmax()]
    
    if(label=='Sem Mascara'):
        image = cv2.rectangle(frame, start_point, end_point, (0,0,255), thickness)
        cv2.putText(image,label,(30,60),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),3)
    if(label=='Com Mascara'):
        image = cv2.rectangle(frame, start_point, end_point, (0,255,0), thickness)
        cv2.putText(image,label,(30,60),cv2.FONT_HERSHEY_SIMPLEX,1.6,(0,0,0),3)
    
    cv2.imshow('preview',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyWindow("preview")