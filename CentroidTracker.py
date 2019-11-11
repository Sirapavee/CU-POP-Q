from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        #self.peopleCounter = 0

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        #self.peopleCounter += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        #if rects is empty no update happen
        if len(rects)==0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects
        
        #compute centroid
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX+endX)/2)
            cY = int((startY+endY)/2)
            inputCentroids[i] = (cX, cY)
        
        #register each input centroid as currently not tracking
        if len(self.objects)==0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
                
        #match input centroid to existing object centroids as currently tracking
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            #compute Euclidean distance
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            
            #Achieve lowest corresponding distance
            #matching (1) find lowest value in each row then sort (2) sort row index based on minimum value 
            rows = D.min(axis=1).argsort()
            #find lowest value in each column and then sort using previously computed row index list
            cols = D.argmin(axis=1)[rows]

            #to see which row and column that already examine
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                #continue if examined
                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.maxDistance:
                    continue
                
                #grab objectID for current row set new centroid and reset disappered counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                #indicate that examined
                usedRows.add(row)
                usedCols.add(col)

            #compute both row and column index that not yet examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            #check if some objects have potentially disappeared
            if D.shape[0] >= D.shape[1]:
                    
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects