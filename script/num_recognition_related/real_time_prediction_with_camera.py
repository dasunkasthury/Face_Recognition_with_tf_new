import cv2
import numpy as np
import tensorflow as tf
import time

def prediction_result(prediction):
  print(" --- ")
  labels = ["Add","Div","Eight","Five","Four","Minus","Multiply","Nine","One","Seven","Six","Three","Two"]
  print("Lite Model Predictions :", labels[prediction], "\n")

def logger(_name, _value):
  print(" ------------ ")
  print(">>", _name, " : ", _value)

def main():
  interpreter = tf.lite.Interpreter('model/vc_model.tflite')
  
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  #video_feed = cv2.VideoCapture('Videos/9.avi') # to get video feed from a specific file
  video_feed = cv2.VideoCapture(0) # get video from the camera realtime

  current_prediction = 0
  previous_prediction = 0

  while(1):
      _, frame = video_feed.read()
      gray_scaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converting image to gray
      cv2.rectangle(gray_scaled, (14,219),(172,363),(255,0,0), 2) # selecting a rectangle in the screen where only this part will be used by the model to predict

      sqaure_region = gray_scaled[219:363, 14:172]

      cv2.imshow("full frame", gray_scaled)
      cv2.imshow("selected_suaure_region", sqaure_region)
      
      circle_loc = cv2.HoughCircles(sqaure_region, cv2.HOUGH_GRADIENT, 1.20,20) # finding the half circles

      # logger("Circle location", circle_loc)
      roi = 0
      cv2.waitKey(1)

      if circle_loc is not None: # if found a circle
        circle_loc = np.round(circle_loc[0, :]).astype("int")
        for i in circle_loc: # find the location of the circle
          center_x = i[0]
          center_y = i[1]
          radius = i[2]
          x = center_x - radius
          y = center_y - radius
          h = 2*radius
          w = 2*radius
          if y > 0 and x > 0:
            cropped_circle = sqaure_region[y:y+h, x:x+w]
            cropped_circle_resize = cv2.resize(cropped_circle, (500, 500))
            roi = cropped_circle_resize[90:390, 90:390]

            input_image = cv2.resize(roi, (128, 128))
            input_image = np.expand_dims(input_image, axis=2) # model is accepting [1,128,128,1] shape but the image generated is [128,128] so have to add dimentions
            input_image = np.expand_dims(input_image, axis=0)

            input_image = input_image.astype(np.float32) # Have to modify
            # <------------- model is exepting float32 but our image generated is only UINT8 so have to modify the model to 
            # accept UINT8 inputs so that it can predict fast

# this is to check the time taken for each image to predict
            start_time = time.time()
            #applying model
            interpreter.allocate_tensors()
            interpreter.set_tensor(input_details[0]['index'], input_image)
            interpreter.invoke()

            #prediction
            tflite_prediction_result = interpreter.get_tensor(output_details[0]['index'])

            end_time = time.time()
            # logger("Time Taken", end_time-start_time) # this will give time taken for the prediction of each

            # write predictions
            # prediction_result(tflite_prediction_result)  # this will print the prediction every time
            # result = np.argmax(tflite_prediction_result)
            # print(result)
            current_prediction = np.argmax(tflite_prediction_result)
            if not current_prediction == previous_prediction:
              prediction_result(current_prediction)
              previous_prediction = current_prediction


            
            cv2.imshow("Region of Interest", roi)

            cv2.waitKey(1)

if __name__ == '__main__':
  main()