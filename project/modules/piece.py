import numpy as np


class Piece:
    def __init__(self, data):
        self.original = data
        self.data = self.original.copy()
        
        self.top = None
        self.right = None
        self.bottom = None
        self.left = None
        
        self.top_affinity = np.inf
        self.right_affinity = np.inf
        self.bottom_affinity = np.inf
        self.left_affinity = np.inf
        self.score = np.inf
    
    def get_neighbors(self):
        neighbors = []
        if self.top is not None:
            neighbors.append(self.top[0])
        if self.right is not None:
            neighbors.append(self.right[0])
        if self.bottom is not None:
            neighbors.append(self.bottom[0])
        if self.left is not None:
            neighbors.append(self.left[0])
        return neighbors
    
    def rotate(self, k, update_neighbors=False):
        self.data = np.rot90(self.original, k=k)
        
        if update_neighbors:
            if k==1:
                self.left, self.bottom, self.right, self.top = self.top, self.left, self.bottom, self.right
                self.left_affinity, self.bottom_affinity, self.right_affinity, self.top_affinity = self.top_affinity, self.left_affinity, self.bottom_affinity, self.right_affinity
            elif k==2:
                self.bottom, self.right, self.top, self.left = self.top, self.left, self.bottom, self.right
                self.bottom_affinity, self.right_affinity, self.top_affinity, self.left_affinity = self.top_affinity, self.left_affinity, self.bottom_affinity, self.right_affinity
            elif k==3: 
                self.right, self.top, self.left, self.bottom = self.top, self.left, self.bottom, self.right
                self.right_affinity, self.top_affinity, self.left_affinity, self.bottom_affinity = self.top_affinity, self.left_affinity, self.bottom_affinity, self.right_affinity
