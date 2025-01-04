import cv2
import numpy as np

def logger(_name, _value):
  print(" ------------ ")
  print(">>", _name, " : ", _value)

def main():
  video_feed = cv2.VideoCapture('Videos/plus.avi')
  iterator = 0

  while(1):
    _,frame = video_feed.read()
    gray_scale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    circle_loc = cv2.HoughCircles(gray_scale, cv2.HOUGH_GRADIENT, 1.20,20)
    logger("Circle location", circle_loc)
    roi = 0
    roi_resized = 0
    number = 0
    if circle_loc is not None:
      circle_loc = np.round(circle_loc[0, :]).astype("int")
      for i in circle_loc:
        center_x = i[0]
        center_y = i[1]
        radius = i[2]
        x = center_x - radius
        y = center_y - radius
        h = 2*radius
        w = 2*radius

        if x>0 and y>0 :
          roi = gray_scale[y:y+h, x:x+w]
          roi_resized = cv2.resize(roi,(500,500))
          number = roi_resized[90:390, 90:390]

    


    iterator = iterator + 1
    cv2.imwrite("Extracted_imges/Add/"+str(iterator)+'.jpg', number)
    cv2.imshow("camera_feed", gray_scale)
    cv2.imshow("Region of Interest", number)
    cv2.waitKey(1)

if __name__ == '__main__':
  main()