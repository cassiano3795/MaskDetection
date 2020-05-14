from tensorflow.keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

classifier = load_model('./testemodel.h5')

class_labels = ['Sem Mascara', 'Com Mascara']
start_point = (15, 15)
end_point = (400, 80) 
thickness = -1
scale_factor = 1.3

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

while True:
    ret, frame = vc.read()

    gray = cv2.resize(frame,(224,224))
    roi = gray.astype('float')/255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis=0)

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