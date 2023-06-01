import numpy as np
import networkx as nx

from itertools import combinations
from .metric import compute_affinity
from .solvers import solver_3x3, solver_3x4, solver_4x4


class Puzzle:
    def __init__(self, pieces):
        if len(pieces) not in [9, 12, 16]:
            raise ValueError(f'Invalid number of pieces, puzzle should be composed of 9, 12 or 16 pieces but got {len(pieces)} pieces.')
        
        self.pieces = pieces
        
        if len(pieces) == 9:
            self.data = np.zeros((384, 384, 3), dtype=pieces[0].data.dtype)
            self.size = (3, 3)
        
        elif len(pieces) == 12:
            self.data = np.zeros((384, 512, 3), dtype=pieces[0].data.dtype)
            self.size = (3, 4)
        
        elif len(pieces) == 16:
            self.data = np.zeros((512, 512, 3), dtype=pieces[0].data.dtype)
            self.size = (4, 4)
    
    def init_affinities(self, crop=None):
        pairs = list(combinations(range(len(self.pieces)), 2))
        permuted = [(pair[1], pair[0]) for pair in pairs]
        pairs.extend(permuted)

        for pair in pairs:
            central = self.pieces[pair[0]]
            side = self.pieces[pair[1]]
            
            for rotation in [0, 1, 2, 3]:
                top_affinity = compute_affinity(central, side, 'top', rotation, crop=crop)
                right_affinity = compute_affinity(central, side, 'right', rotation, crop=crop)
                bottom_affinity = compute_affinity(central, side, 'bottom', rotation, crop=crop)
                left_affinity = compute_affinity(central, side, 'left', rotation, crop=crop)
                
                if top_affinity < central.top_affinity:
                    central.top_affinity = top_affinity
                    central.top = (pair[1], rotation)
                
                if right_affinity < central.right_affinity:
                    central.right_affinity = right_affinity
                    central.right = (pair[1], rotation)
                
                if bottom_affinity < central.bottom_affinity:
                    central.bottom_affinity = bottom_affinity
                    central.bottom = (pair[1], rotation)
                
                if left_affinity < central.left_affinity:
                    central.left_affinity = left_affinity
                    central.left = (pair[1], rotation)
        
        for piece in self.pieces:
            piece.score = piece.top_affinity + piece.right_affinity + piece.bottom_affinity + piece.left_affinity
    
    def get_undirected_graph(self):
        directed_graph = nx.DiGraph()
        for i, piece in enumerate(self.pieces):
            directed_graph.add_node(i, score=piece.score)
        
        for i, piece in enumerate(self.pieces):
            directed_graph.add_edge(i, piece.top[0])
            directed_graph.add_edge(i, piece.right[0])
            directed_graph.add_edge(i, piece.bottom[0])
            directed_graph.add_edge(i, piece.left[0])
            
        undirected_graph = nx.Graph()
        for i, piece in enumerate(self.pieces):
            undirected_graph.add_node(i, score=piece.score)
            
        for edge in directed_graph.edges():
            u, v = edge

            # Check if both (u, v) and (v, u) exist in the directed graph
            if directed_graph.has_edge(u, v) and directed_graph.has_edge(v, u):
                undirected_graph.add_edge(u, v)
        
        self.graph = undirected_graph
        return undirected_graph
    
    def solve(self):
        self.init_affinities()
        G = self.get_undirected_graph()

        if self.size == (3, 3):
            self.pieces = solver_3x3(self, G)
        elif self.size == (3, 4):
            self.pieces = solver_3x4(self, G)
        elif self.size == (4, 4):
            self.pieces = solver_4x4(self, G)
    
    def assemble(self):
        for idx, piece in enumerate(self.pieces):
            i = idx // self.size[0]
            j = idx % self.size[1]
            self.data[128*i:128*(i+1), 128*j:128*(j+1), :] = piece.data
    
    def random_permutation(self):
        self.pieces = np.random.permutation(self.pieces)
    
    def random_rotations(self):
        for piece in self.pieces:
            piece.rotate(np.random.randint(4))
            piece.original = piece.data.copy()