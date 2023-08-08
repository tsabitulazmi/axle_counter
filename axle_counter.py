from this import d
import cv2
import time
import math
import numpy as np

# Set region of interest
x1 = 300
x2 = 1200
y1 = 400
y2 = 800
reg_ids = set()

# Video source
src = 'cars_passing_input.mp4'

# Import haar cascade
cascade_src = 'axle.xml'
axle_cascade = cv2.CascadeClassifier(cascade_src)

# Initialize count
count = 0
fcount = 0
center_points_prev_frame = []
tracking_objects = {}
track_id = 0
center_points_cur_frame = []
gol12 = 0
gol3 = 0
gol4 = 0
gol5 = 0
axle_count_prev = 100

# Time frame
timeframe = time.time()
frame_id = 0

cap = cv2.VideoCapture(src)

while True:
    ret, frame = cap.read()
    frame_id +=1
    count +=1
    fcount +=1

    if (type(frame) == type(None)):
        break

    # Extract Region of interest
    height, width, channels = frame.shape
    roi = frame[y1:y2, x1:x2]
    reg = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    axle = axle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))

    detections = []

    for (x,y,w,h) in axle:
        cv2.rectangle(roi,(x,y),(x+w,y+h),(255,0,0),3) 
        #cv2.putText(img, 'car', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)
        xmid = int((x + x+w)/2)
        ymid = int((y + y+h)/2)
        #cv2.circle(roi,(xmid,ymid),5,(0,0,255),-1)
        center_points_cur_frame.append([xmid,ymid])

     # Only at the beginning we compare previous and current frame
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 350:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 350:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.circle(roi, pt, 3, (0, 0, 255), -1)
        cv2.putText(roi, str(object_id+1), (pt[0], pt[1]), 0, 2, (0, 0, 255),2)
        inside_region = cv2.pointPolygonTest(np.array(reg), (pt[0],pt[1]), False)
        if(inside_region < 0):
            reg_ids.add(object_id)
            fcount = 0

    axle_count = len(reg_ids)

    if (fcount == 5 and axle_count == axle_count_prev):
        reg_ids.clear()
        if(axle_count == 2):
            gol12 += 1
        elif(axle_count == 3):
            gol3 += 1
        elif(axle_count == 4):
            gol4 += 1
        elif(axle_count == 5):
            gol5 += 1

    axle_count_prev = axle_count

    # Counting fps
    elapsedtime = time.time() - timeframe
    fps = round(frame_id/ elapsedtime , 2)

    cv2.polylines(frame, [np.array(reg)], True, (0,255,255),3)
    cv2.rectangle(frame, (0,0), (350,150), (255,255,255), -1)
    cv2.putText(frame, 'Count: ' + str(axle_count) + '   FPS : ' + str(fps), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0))
    cv2.putText(frame, 'Gol 1,2 : ' + str(gol12), (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0))
    cv2.putText(frame, 'Gol 3   : ' + str(gol3), (10,85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0))
    cv2.putText(frame, 'Gol 4   : ' + str(gol4), (10,115), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0))
    cv2.putText(frame, 'Gol 5   : ' + str(gol5), (10,145), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0))

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()