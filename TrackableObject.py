class TrackableObject:
    def __init__(self, objID, centroid):
        self.objID = objID
        self.centroids = [centroid]
        self.counted = False
        self.waited = False