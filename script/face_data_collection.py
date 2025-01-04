import os
import time
import uuid
import cv2

# ----------------------------------------------------------------- Collect Images  --------------------------------------------------------------------------------------
IMAGES_PATH = 'data/extracted_images'
number_images = 80

cap = cv2.VideoCapture(0)
for imgnum in range(number_images):
    print('Collecting image {}'.format(imgnum))
    ret, frame = cap.read()
    imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# #----------------------------------------------------------------- Create Dataset ----------------------------------------------------------------- 
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus: 
#     tf.config.experimental.set_memory_growth(gpu, True)

# def load_image(x): 
#     byte_img = tf.io.read_file(x)
#     img = tf.io.decode_jpeg(byte_img)
#     return img

# #----------------------------------------------------------------- Move Matching Labels AFTER Annotation with Labelme ----------------------------------------------------------------- 


# for folder in ['train','test','val']:
#     for file in os.listdir(os.path.join('data', folder, 'images')):
        
#         filename = file.split('.')[0]+'.json'
#         existing_filepath = os.path.join('data','labels', filename)
#         if os.path.exists(existing_filepath): 
#             new_filepath = os.path.join('data',folder,'labels',filename)
#             os.replace(existing_filepath, new_filepath)  


# # ----------------------------------------------------------------- Image Augmentation ----------------------------------------------------------------- 

# augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), 
#                          alb.HorizontalFlip(p=0.5), 
#                          alb.RandomBrightnessContrast(p=0.2),
#                          alb.RandomGamma(p=0.2), 
#                          alb.RGBShift(p=0.2), 
#                          alb.VerticalFlip(p=0.5)], 
#                         keypoint_params=alb.KeypointParams(format='xy', label_fields=['class_labels']))


# for partition in ['train', 'test', 'val']: 
#     for image in os.listdir(os.path.join('data', partition, 'images')):
#         img = cv2.imread(os.path.join('data', partition, 'images', image))

#         classes = [0,0]
#         coords = [0,0,0.00001,0.00001]
#         label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
#         if os.path.exists(label_path):
#             with open(label_path, 'r') as f:
#                 label = json.load(f)
    
#             if label['shapes'][0]['label']=='LeftEye': 
#                 classes[0] = 1
#                 coords[0] = np.squeeze(label['shapes'][0]['points'])[0]
#                 coords[1] = np.squeeze(label['shapes'][0]['points'])[1]

#             if label['shapes'][0]['label']=='RightEye':
#                 classes[1] = 1
#                 coords[2] = np.squeeze(label['shapes'][0]['points'])[0]
#                 coords[3] = np.squeeze(label['shapes'][0]['points'])[1]

#             if len(label['shapes']) > 1:     
#                 if label['shapes'][1]['label'] =='LeftEye': 
#                     classes[0] = 1 
#                     coords[0] = np.squeeze(label['shapes'][1]['points'])[0]
#                     coords[1] = np.squeeze(label['shapes'][1]['points'])[1]

#                 if label['shapes'][1]['label'] =='RightEye': 
#                     classes[1] = 1
#                     coords[2] = np.squeeze(label['shapes'][1]['points'])[0]
#                     coords[3] = np.squeeze(label['shapes'][1]['points'])[1]
            
#             np.divide(coords, [640,480,640,480])
                
#         try: 
#             for x in range(120):
#                 keypoints = [(coords[:2]), (coords[2:])]
#                 augmented = augmentor(image=img, keypoints=keypoints, class_labels=['LeftEye','RightEye'])
#                 cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

#                 annotation = {}
#                 annotation['image'] = image
#                 annotation['class'] = [0,0]
#                 annotation['keypoints'] = [0,0,0,0]

#                 if os.path.exists(label_path):
#                     if len(augmented['keypoints']) > 0: 
#                         for idx, cl in enumerate(augmented['class_labels']):
#                             if cl == 'LeftEye': 
#                                 annotation['class'][0] = 1 
#                                 annotation['keypoints'][0] = augmented['keypoints'][idx][0]
#                                 annotation['keypoints'][1] = augmented['keypoints'][idx][1]
#                             if cl == 'RightEye': 
#                                 annotation['class'][1] = 1 
#                                 annotation['keypoints'][2] = augmented['keypoints'][idx][0]
#                                 annotation['keypoints'][3] = augmented['keypoints'][idx][1]
                                
#                 annotation['keypoints'] = list(np.divide(annotation['keypoints'], [450,450,450,450]))


#                 with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
#                     json.dump(annotation, f)

#         except Exception as e:
#             print(e)