import cv2
from flask import Flask, render_template, Response
import numpy as np
import pytesseract
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'

table_data = []  # List to store table data

def matr():
    path = r'Mat_data\vids\20.mp4'
    cap = cv2.VideoCapture(path)
    licence_cascade = cv2.CascadeClassifier(r"Mat_data\haarcascade_russian_plate_number.xml")
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        licences = licence_cascade.detectMultiScale(gray, 1.1, 6)

        for (x, y, w, h) in licences:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            imroi = frame[y:y+h, x:x+w]
            text = pytesseract.image_to_string(imroi, lang='eng', config='--psm 6')
            txt = ''.join([c for c in text if c.isalnum() or c == '-' or c == '|'])
            cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            if 6 < len(txt) < 10 and txt.isupper():
                cv2.imwrite(r"frontend\Mat_data\mat/" + txt + ".jpg", imroi)
                timestamp = datetime.now().strftime('%H:%M %B %Y')  # Get current timestamp
                table_data.append((txt, txt + ".jpg", timestamp))  # Store table data
                

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
