import cv2 #opencv-python==4.10.0.84
import numpy as np #numpy==1.26.4
import tensorflow as tf #tensorflow==2.17.0


cap = cv2.VideoCapture(0)
# facetracker = load_model('trained_models/facetracker_40h.h5')
facetracker = tf.keras.models.load_model('trained_model/facetracker_largedata_40h.h5')
while cap.isOpened():
    _ , frame = cap.read()
    frame = frame[50:500, 50:500,:]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    
    yhat = facetracker.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]
    
    if yhat[0] > 0.5: 
        # Controls the main rectangle
        cv2.rectangle(frame, 
                    tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                    tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), (255,0,0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                    tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), [0,-30])),
                    tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),[80,0])), (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),[0,-5])),cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow('EyeTrack', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()