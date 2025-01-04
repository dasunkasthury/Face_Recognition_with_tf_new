import cv2 #opencv-python==4.10.0.84
import numpy as np #numpy==1.26.4
import tensorflow as tf #tensorflow==2.17.0

from firebase_helper import updateDb

import time
import torch


cap = cv2.VideoCapture(0)
facetracker = tf.keras.models.load_model('trained_model/facetracker_largedata_40h.h5')
faceEmotionTracker = tf.keras.models.load_model('trained_model/New_model_1/face_recognition_model_5_17E_128BS_1500SPE_200VS_0.62A.h5') # cahnge the model here 

dowsiness_model = torch.hub.load('ultralytics/yolov5', 'custom', path='trained_model/YOLO/last.pt', force_reload=True)

current_prediction = ''
previous_prediction = ''

def prediction_emotion(prediction):
  labels = ["angry","disgust","fear","happy","neutral","sad","suprise"]

  if isinstance(prediction, str):
      try:
        prediction = int(prediction)
      except ValueError:
        raise ValueError("Prediction must be an integer or a string that can be converted to an integer.")
  
  # Check if the prediction is within the valid range
  if 0 <= prediction < len(labels):
    return labels[prediction]
  else:
    raise IndexError("Prediction index out of range.")
  

while cap.isOpened():
    _ , frame = cap.read()

    
    # -- YOLO implementation

    results = dowsiness_model(frame)
    # -- END YOLO implementation

    frame = frame[50:500, 50:500,:]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # print("Original shape of rgb:", rgb.shape)
    resized = tf.image.resize(rgb, (120,120))
    yhat_face_frame = facetracker.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat_face_frame[1][0] 
    rio_coordinates = np.array(sample_coords)
    scaled_roi_coordinates = rio_coordinates * 450
    x_start, y_start = map(int, scaled_roi_coordinates[:2])
    x_end, y_end = map(int, scaled_roi_coordinates[2:])
    if (x_start > 0 and y_start > 0 and x_end > 0 and y_end > 0):
      roi_face_region = frame[y_start:y_end, x_start:x_end]
    else:
      roi_face_region = frame[10:30, 10:30]
    # prediction of the facial emotion ------------------------------------------------------------
    gray_image = tf.image.rgb_to_grayscale(roi_face_region)
    gray_frame = cv2.cvtColor(roi_face_region, cv2.COLOR_BGR2GRAY)


    # resized_gray = tf.image.resize(gray_image, (48,48)) # change the trained model input image size to match
    resized_gray = tf.image.resize(gray_image, (48,48)) # change the trained model input image size to match

    yhat_emotion = faceEmotionTracker.predict(np.expand_dims(resized_gray/255,0))

    predictedEmotion = np.argmax(yhat_emotion)

    current_prediction = prediction_emotion(predictedEmotion)
    if not current_prediction == previous_prediction:
      # update_variable_if_unchanged(current_prediction, previous_prediction)
      updateDb("mood", current_prediction)
      # Update DB
      # if current_prediction == "happy":
      #   updateDb("bright", 255)
      # else:
      #   updateDb("bright", 0)
      previous_prediction = current_prediction

    
    if yhat_face_frame[0] > 0.5: 
        # Controls the main rectangle
        cv2.rectangle(frame, 
                    tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                    tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), (255,0,0), 2)
        
        # Controls the label rectangle to high light the predicted value
        cv2.rectangle(frame, 
                    tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), [0,-30])),
                    tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),[80,0])), (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, current_prediction, tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),[0,-5])),cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255,255,255), 2, cv2.LINE_AA)
        

        
    
    current_predict = current_prediction
    
    # cv2.imshow('FaceTrack', gray_frame)
    # cv2.imshow('Full', frame)
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


