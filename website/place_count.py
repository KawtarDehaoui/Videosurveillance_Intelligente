import cv2
import matplotlib.pyplot as plt
import numpy as np
from .util import get_parking_spots_boxes,empty_or_not

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


def places():
 
 mask = r"website\data\data\mask.png"
 video_path = r"website\data\data\parking_loop.mp4"
 

 mask = cv2.imread(mask , 0)

 cap = cv2.VideoCapture(video_path)

 connected_comp = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

 spots = get_parking_spots_boxes(connected_comp)

 spots_status = [None for j in spots]
 diffs = [None for j in spots]

 previous_frame = None
 frame_num = 0

 print(spots[0])
 ret = True
 step = 30
 while ret :
    ret, frame = cap.read()

    if frame_num % step == 0 and  previous_frame is not None:
        for spot_index, spot in enumerate(spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs[spot_index] = calc_diff(spot_crop ,previous_frame[y1:y1 + h, x1:x1 + w, :])

        

    if frame_num % step == 0:
        if previous_frame is None:
            arr = range(len(spots))
        else:
            arr = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_index in arr:
            spot = spots[spot_index]
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            spot_status = empty_or_not(spot_crop)

            spots_status[spot_index] = spot_status
    if frame_num % step == 0:
        previous_frame = frame.copy()
    for spot_index, spot in enumerate(spots):
        spot_status = spots_status[spot_index]
        x1, y1, w, h = spots[spot_index]

        if spot_status:
            frame = cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (0,255,0), 2)
        else:
            frame = cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (0,0,255), 2)



    cv2.putText(frame, 'Available places: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)



    
    re,buffer=cv2.imencode('.jpg',frame)
    frame=buffer.tobytes()

    yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    frame_num += 1

