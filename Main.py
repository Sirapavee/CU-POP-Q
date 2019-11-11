import cv2 as cv
import numpy as np
import imutils
from CentroidTracker import CentroidTracker
from TrackableObject import TrackableObject

cap = cv.VideoCapture('videos/example_01.mp4')
#out = cv.VideoWriter('output/result.avi', cv.VideoWriter_fourcc('M','J','P','G'), 30, (640,480))
ct = CentroidTracker()
bgs = cv.createBackgroundSubtractorMOG2(200, 50, False)

trackableObjects = {}

#frame width and height
W = None
H = None

totalIn = 0
totalOut = 0
totalWait = 0

while True:
    ret, frame = cap.read()

    #video end
    if frame is None:
        break

    frame = imutils.resize(frame, 400)

    if W is None and H is None:
        (H, W) = frame.shape[:2]

    rects = []
    
    mask = bgs.apply(frame)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, (5,5), iterations=3)
    mask = cv.morphologyEx(mask, cv.MORPH_DILATE, (5,5), iterations=6)
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    try: hierarchy = hierarchy[0]
    except: hierarchy = []

    #detect people
    for cntr, heir in zip(contours, hierarchy):
        if cv.contourArea(cntr)>1050:
            (x, y, w, h) = cv.boundingRect(cntr)
            rects.append((x, y, x+w, y+h))
            '''cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)
            cv.circle(frame, (x+w//2,y+h//2), 1, (0,0,255), 5)'''
    
    #draw wait detection line
    cv.line(frame, (0, int(H*0.3)), (W, int(H*0.3)), (0,0,255), 1)
    #draw in and out detection line
    cv.line(frame, (0, int(H*0.7)), (W, int(H*0.7)), (0,0,255), 1)
    
    objects = ct.update(rects)
    
    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)

        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            #check if wait
            if not to.waited:
                if direction > 0 and centroid[1] > int(H*0.3):
                    totalWait +=1
                    to.waited = True
                    
            #count as in/out from bus
            if not to.counted:
                if direction < 0 and centroid[1] < int(H*0.7): #or H//2
                    totalOut += 1
                    to.counted = True
                    
                    #get out no one wait for bus anymore
                    if not to.waited:
                        totalWait -= 0
                elif direction > 0 and centroid[1] > int(H*0.7): #or H//2
                    totalIn += 1
                    to.counted = True
                    totalWait -= 1

        trackableObjects[objectID] = to

        text = 'ID {}'.format(objectID)
        cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    info = [('Out', totalOut), ('In', totalIn), ('WAIT', totalWait)]
    for (i, (k,v)) in enumerate(info):
        text = '{}: {}'.format(k,v)
        cv.putText(frame, text, (10, H - ((i * 20) + 20)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    #write new video
    #out.write(frame)

    #showing result
    cv.imshow('frame', frame)
    if cv.waitKey(10) & 0xFF == ord('x'):
        break

cap.release()
#out.release()
cv.destroyAllWindows()