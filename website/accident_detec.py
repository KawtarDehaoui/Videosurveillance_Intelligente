import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from flask import Blueprint, render_template, request, flash, redirect, url_for,Response
import winsound

   
# Load the trained model

model_dir = r"accident_data"
loaded_model = tf.keras.models.load_model(os.path.join(model_dir, "trained_model.h5"))

# Load the class names
class_names_df = pd.read_csv(os.path.join(model_dir, "class_acc.csv"))
class_names = class_names_df.iloc[0].to_dict()



def predict_frame(img):
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = (loaded_model.predict(img_batch) > 0.5).astype("int32")
    if(prediction[0][0]==1):
        return("Accident Detected")
    else:
        return("No Accident")


def accident():

    path = r"accident_data\IMG_8562.mp4"
    cap = cv2.VideoCapture(path)
   
    img_height = 48
    img_width = 48

    skip_frames = 10  # Number of frames to skip
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        if frame_counter % skip_frames != 0:
            continue  # Skip the frame if it's not the frame to be processed

        resized_frame = cv2.resize(frame, (img_width, img_height))
        resized_frame_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        prediction = predict_frame(resized_frame_gray)

        if prediction == "Accident Detected":
            cv2.putText(frame, "Accident Detected", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)
            winsound.PlaySound(r'frontend\accident.mp3', winsound.SND_ASYNC)
        re, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
