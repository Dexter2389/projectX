### ---------- Handle Command Line Arguments ----------

import argparse

hide_img = False # default is to display image after predictions

a = argparse.ArgumentParser(description="Predict the class of a given driver image.")
a.add_argument("--folder", help="path to folder", default="/dataset/test_images")
a.add_argument("--hide_img", action="store_true", help="do NOT display image on prediction termination")
args = a.parse_args()

if args.hide_img:
    hide_img = True
    
if args.folder is not None:
    dir_path = args.folder

### ---------- Import Relevant Libraries ----------

from keras.preprocessing.image import load_img, img_to_array
from helper import create_top_model, class_labels, target_size
import numpy as np
import cv2
from imutils.video import VideoStream
from keras import applications
import operator
import matplotlib.pyplot as plt
import os
import cv2
import av

dir_path = args.folder

for filename in os.listdir(dir_path):
    # prepare image for classification using keras utility functions
    img = cv2.imread(dir_path+'/'+filename)
    image = load_img(dir_path+'/'+filename, target_size=target_size)
    print(image)

    image_arr = img_to_array(image) # convert from PIL Image to NumPy array
    image_arr /= 255

    # to be able to pass it through the network and use batches, we want it with shape (1, 224, 224, 3)
    image_arr = np.expand_dims(image_arr, axis=0)

    # build the VGG16 network  
    model = applications.VGG16(include_top=False, weights='imagenet')  

    # get the bottleneck prediction from the pre-trained VGG16 model  
    bottleneck_features = model.predict(image_arr) 

    # build top model  
    model = create_top_model("softmax", bottleneck_features.shape[1:])

    model.load_weights("/src/res/_top_model_weights.h5")

    predicted = model.predict(bottleneck_features)
    decoded_predictions = dict(zip(class_labels, predicted[0]))
    decoded_predictions = sorted(decoded_predictions.items(), key=operator.itemgetter(1), reverse=True)

    print()
    x_coor = 100
    y_coor = 1500
    text_color = (100, 100, 200)
    count = 1

    for key, value in decoded_predictions[:5]:        
        font = cv2.FONT_HERSHEY_SIMPLEX
        z = cv2.putText(img,"{}. {}: {:8f}%".format(count, key, value*100), (x_coor,y_coor), font, 1,(255,255,255),2,cv2.LINE_AA)
        print("{}. {}: {:8f}%".format(count, key, value*100))
        count += 1
        y_coor += 60

    cv2.imwrite(dir_path+'/'+'test_final_'+filename,z)
    
    
    if not hide_img:
        cv2.imshow(z)
    
    else:
        cv2.imwrite(dir_path+'/'+'test_final_'+filename,z)

cv2.waitKey(0)
cv2.destroyAllWindows()
